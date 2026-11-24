# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SBTO005 with teacher training dynamics and distilled student observations."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.array.env import EnvObs, FinalObs
from motrix_env_core.observations import ObservationGroupSpec, ObservationSpec
from motrix_envs.locomotion.wbt.g1.sbto005_distill.camera import (
    DEPTH,
    HEIGHT,
    WIDTH,
    attach_depth_camera,
    isolate_head_for_rays,
)
from motrix_envs.locomotion.wbt.g1.sbto005_distill.depth_noise import DepthNoise
from motrix_envs.locomotion.wbt.g1.sbto005_distill.policy import STATE_SIZE, student_terms
from motrix_envs.locomotion.wbt.g1.sbto005_distill.visibility import VisibleBoxEnv, configure_visible_box
from motrix_envs.locomotion.wbt.g1.sbto005_teacher import Sbto005TeacherCfg, make_sbto005_teacher_cfg

TASK = "g1-29dof-wbt-largebox-sbto005-student"


@registry.envcfg(TASK)
def make_sbto005_student_cfg() -> Sbto005TeacherCfg:
    cfg = make_sbto005_teacher_cfg()
    fixed_box = os.environ.get("MOTRIX_SBTO005_STUDENT_FIXED_BOX") == "1"
    if fixed_box:
        cfg.commands.motion.start_at_timestep_zero_prob = 0.0
    configure_visible_box(cfg, fixed_box=fixed_box)
    return cfg


@registry.env(TASK)
class Sbto005StudentEnv(VisibleBoxEnv):
    def __init__(self, cfg, *args, **kwargs):
        with TemporaryDirectory(prefix="sbto-student-") as directory:
            model = Path(directory) / "g1.xml"
            isolate_head_for_rays(Path(cfg.scene.objs.robot.model.file), model)
            source = cfg.scene.objs.robot.model.file
            attach_depth_camera(cfg, robot_mjcf=model)
            try:
                super().__init__(cfg, *args, **kwargs)
            finally:
                cfg.scene.objs.robot.model.file = source
        teacher_spec = self.observation_manager.spec
        policy_space = gym.spaces.Box(-np.inf, np.inf, (STATE_SIZE + HEIGHT * WIDTH,), dtype=np.float32)
        self._student_spec = ObservationSpec(teacher_spec.groups | {"policy": ObservationGroupSpec(policy_space)})
        self._state_columns = None
        self._depth_device = torch.device("cpu")
        self._depth_noise = None

    @property
    def observation_spec(self):
        return self._student_spec

    def set_depth_device(self, device: torch.device) -> None:
        self._depth_device = device

    def init_state(self):
        self._depth_noise = DepthNoise(self.num_envs, self._depth_device, self._build_seed + 89471)
        return super().init_state()

    def _build_task_program(self):
        program = super()._build_task_program()
        self._state_columns = student_terms(self)
        return program

    def _pack_student_obs(self, observations, depth):
        state = observations.policy.index_select(1, self._state_columns)
        packed = torch.cat((state, depth.flatten(1).to(state.device, dtype=state.dtype)), dim=1)
        return EnvObs(observations.groups | {"policy": packed})

    def compute_observation(self, state):
        super().compute_observation(state)
        fresh = torch.nonzero(state.episode_steps == 0).flatten()
        protect = fresh[self._episode_start_frame[fresh] == 0]
        depth = self._depth_noise.frame(self.sim_data[DEPTH], fresh, self.box_pixel_mask, 1.0, protect=protect)
        return state.replace(obs=self._pack_student_obs(self.observation_manager.observation, depth))

    def _capture_final_obs(self, env_ids):
        terminal = super()._capture_final_obs(env_ids)
        depth = self._depth_noise.preview(self.sim_data[DEPTH][env_ids], env_ids, 1.0)
        return FinalObs(terminal.env_ids, self._pack_student_obs(terminal.obs, depth))
