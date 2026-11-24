# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Own pre-step student observations and label them with the frozen teacher."""

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from motrix_rl_builtin.fastsac.wrap import FastSacEnvWrap

from motrix_env_core.sim.read import LinkPositionQuery, LinkQuaternionQuery
from motrix_envs.locomotion.wbt.g1.sbto005_distill.camera import (
    DEPTH,
    HEIGHT,
    WIDTH,
    attach_depth_camera,
    isolate_head_for_rays,
    student_depth,
)
from motrix_envs.locomotion.wbt.g1.sbto005_distill.depth_noise import RAMP_STEPS, DepthNoise
from motrix_envs.locomotion.wbt.g1.sbto005_distill.policy import ACTION_SIZE, STATE_SIZE, Student, student_terms
from motrix_envs.locomotion.wbt.g1.sbto005_distill.visibility import VisibleBoxEnv, configure_visible_box
from motrix_envs.locomotion.wbt.g1.sbto005_teacher import TASK, Sbto005TeacherEnv, make_sbto005_teacher_cfg
from motrix_rl import runner, runs


class EvaluationEnv(Sbto005TeacherEnv):
    """Keep the first terminal state of each lane available for scoring."""

    def _reset_done_envs(self) -> None:
        if self._initializing_episode_batch:
            super()._reset_done_envs()


class VisibleEvaluationEnv(EvaluationEnv, VisibleBoxEnv):
    """Use the visibility-constrained reset when starting an evaluation batch."""


@dataclass
class Samples:
    state: torch.Tensor
    depth: torch.Tensor
    action: torch.Tensor

    @classmethod
    def allocate(cls, steps: int, width: int, *, shared: bool = False) -> "Samples":
        state = torch.empty(steps, width, STATE_SIZE)
        depth = torch.empty(steps, width, 1, HEIGHT, WIDTH, dtype=torch.float16)
        action = torch.empty(steps, width, ACTION_SIZE)
        if shared:
            for tensor in (state, depth, action):
                tensor.share_memory_()
        return cls(state, depth, action)

    def take(self, steps: int, width: int) -> "Samples":
        return Samples(*(x[:steps, :width].flatten(0, 1) for x in (self.state, self.depth, self.action)))

    def clone(self) -> "Samples":
        return Samples(self.state.clone(), self.depth.clone(), self.action.clone())

    def sample(self, count: int, rng: torch.Generator) -> "Samples":
        return self.index(torch.randint(len(self.state), (count,), generator=rng))

    def index(self, ids: torch.Tensor) -> "Samples":
        return Samples(self.state[ids], self.depth[ids], self.action[ids])

    def to(self, device: torch.device) -> "Samples":
        return Samples(self.state.to(device), self.depth.to(device, dtype=torch.float32), self.action.to(device))


def make_env(
    count: int,
    seed: int,
    *,
    camera: bool = True,
    evaluation: bool = False,
    box_xy: float = 0.10,
    box_yaw: float = 30.0,
    perturbations: bool = True,
    visible_box: bool = False,
) -> Sbto005TeacherEnv:
    torch.manual_seed(seed)
    cfg = make_sbto005_teacher_cfg()
    cfg.commands.motion.adaptive_sampling_enabled = False
    cfg.commands.motion.start_at_timestep_zero_prob = 1.0
    cfg.hand_contact_curriculum = None
    cfg.sim_reset.object_state.initial_xy_noise = box_xy
    cfg.sim_reset.object_state.initial_yaw_noise = box_yaw * (torch.pi / 180)
    if not perturbations:
        duration = cfg.max_episode_seconds
        cfg = cfg.for_play()
        cfg.max_episode_seconds = duration
    if evaluation:
        base = cfg.scene.objs.robot.resolved_base_link_name
        cfg.queries.data.update(
            distill_base_pos=LinkPositionQuery(link=base), distill_base_quat=LinkQuaternionQuery(link=base)
        )
    with TemporaryDirectory(prefix="sbto-distill-") as directory:
        if camera:
            model = Path(directory) / "g1.xml"
            isolate_head_for_rays(Path(cfg.scene.objs.robot.model.file), model)
            attach_depth_camera(cfg, robot_mjcf=model)
        if visible_box:
            if not camera:
                raise ValueError("The visible box reset requires the student's depth camera.")
            configure_visible_box(cfg)
            env_cls = VisibleEvaluationEnv if evaluation else VisibleBoxEnv
        else:
            env_cls = EvaluationEnv if evaluation else Sbto005TeacherEnv
        return env_cls(cfg, num_envs=count, backend="motrixsim", seed=seed, headless=True)


def make_teacher(checkpoint_path: Path, env: Sbto005TeacherEnv, device: torch.device):
    found = runs.find_metadata_for_policy(checkpoint_path)
    if found is None:
        raise ValueError("The Teacher checkpoint needs its run metadata.json and task_config.yaml.")
    run_dir, metadata = found
    if (metadata.env_name, metadata.rllib, metadata.algo) != (TASK, "motrix", "fastsac"):
        raise ValueError(f"Expected a Motrix FastSAC checkpoint for {TASK}.")
    run = runs.open_run_context(run_dir, metadata)
    trainer = runner.create_run_handle(run, cfg_override={"device": str(device)}, play_num_envs=env.num_envs).trainer
    agent = trainer._make_agent(FastSacEnvWrap(env, device))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["global_step"] != 410000:
        raise ValueError(f"Expected the 410000-step Teacher, got {checkpoint['global_step']}.")
    agent.load_state_dict(checkpoint, load_optimizers=False)
    agent.actor.eval().requires_grad_(False)
    agent.obs_normalizer.eval().requires_grad_(False)
    return agent


class Rollout:
    def __init__(self, env: Sbto005TeacherEnv, teacher, device: torch.device, seed: int):
        self.env = env
        self.teacher = teacher
        self.device = device
        self.rng = torch.Generator().manual_seed(seed)
        self.state = env.init_state()
        self.columns = student_terms(env)
        self.student = make_student(env).to(device).eval()
        self.depth_noise = DepthNoise(env.num_envs, device, seed + 89471) if isinstance(env, VisibleBoxEnv) else None
        self.control_steps = 0

    @torch.no_grad()
    def collect(self, destination: Samples, steps: int, width: int, beta: float) -> dict:
        started = perf_counter()
        choices = 0
        resets = 0
        for step in range(steps):
            obs = self.state.obs.policy
            state = obs.index_select(1, self.columns)
            depth = student_depth(self.env.sim_data[DEPTH])
            teacher = self.teacher.act(obs.to(self.device), deterministic=True)
            expert = teacher.cpu()
            rows = (
                slice(None)
                if width == self.env.num_envs
                else torch.randperm(self.env.num_envs, generator=self.rng)[:width]
            )
            destination.state[step, :width].copy_(state[rows])
            destination.depth[step, :width].copy_(depth[rows])
            destination.action[step, :width].copy_(expert[rows])
            if not beta:
                action = self.student(state.to(self.device), depth.to(self.device)).cpu()
            elif beta == 1.0:
                choices += self.env.num_envs
                action = expert
            else:
                use_expert = torch.rand(self.env.num_envs, 1, generator=self.rng) < beta
                choices += int(use_expert.sum())
                prediction = self.student(state.to(self.device), depth.to(self.device)).cpu()
                action = torch.where(use_expert, expert, prediction)
            self.state = self.env.step(action)
            resets += int(self.state.done.sum())
        return {
            "steps": steps * self.env.num_envs,
            "teacher_actions": choices,
            "resets": resets,
            "seconds": perf_counter() - started,
        }

    @torch.no_grad()
    def collect_student(self, destination: Samples) -> dict:
        """Collect one full-batch rollout with Teacher labels buffered on the policy device."""
        started = perf_counter()
        steps, width = destination.state.shape[:2]
        columns = self.columns.to(self.device)
        labels = torch.empty_like(destination.action, device=self.device)
        noise = self.depth_noise
        noisy_replay = torch.empty_like(destination.depth, device=self.device) if noise is not None else None
        resets = 0
        for step in range(steps):
            obs = self.state.obs.policy
            state = obs.index_select(1, self.columns)
            raw_depth = self.env.sim_data[DEPTH]
            if noise is None:
                depth = student_depth(raw_depth)
                destination.depth[step].copy_(depth)
            else:
                fresh = torch.nonzero(self.state.episode_steps == 0).flatten()
                strength = min(1.0, (self.control_steps + step) / RAMP_STEPS)
                depth = noise.frame(raw_depth, fresh, self.env.box_pixel_mask, strength)
                noisy_replay[step].copy_(depth)
            policy_obs = obs.to(self.device)
            labels[step].copy_(self.teacher.act(policy_obs, deterministic=True))
            action = self.student(policy_obs.index_select(1, columns), depth.to(self.device, dtype=torch.float32))
            destination.state[step].copy_(state)
            self.state = self.env.step(action.cpu())
            resets += int(self.state.done.sum())
        self.control_steps += steps
        destination.action.copy_(labels)
        stats = {"steps": steps * width, "teacher_actions": 0, "resets": resets, "seconds": perf_counter() - started}
        if noise is not None:
            destination.depth.copy_(noisy_replay)
            stats["seconds"] = perf_counter() - started
            stats["visibility"] = self.env.visibility_counts | {
                "histogram_xy": self.env.visibility_histogram.tolist(),
                "first_frames": noise.first_frames,
                "first_frame_passes": noise.first_frame_passes,
                "first_frame_mask_fallbacks": noise.first_frame_fallbacks,
            }
            stats["depth_noise_strength"] = min(1.0, self.control_steps / RAMP_STEPS)
        return stats


def make_student(env: Sbto005TeacherEnv) -> Student:
    return Student(torch.from_numpy(env.action_space.low).clone(), torch.from_numpy(env.action_space.high).clone())


def configure_motion(motion: Path) -> None:
    os.environ.update(
        MOTRIX_SBTO005_MOTION=str(motion.resolve()),
        MOTRIX_SBTO005_PLAY_PERTURBATIONS="0",
        MOTRIX_SBTO005_HAND_CONTACT_REWARD="grasp",
        MOTRIX_SBTO005_HAND_CONTACT_WEIGHT="5",
    )
