# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Robot-only carrying tracking with PC2 target clipping and physical perturbations."""

import numpy as np

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.sim import LinkAngularVelocityQuery, LinkLinearVelocityQuery
from motrix_env_core.sim.model import GeomFrictionQuery
from motrix_env_core.sim.write import BodyAngularVelocityWrite, BodyLinearVelocityWrite, GeomFrictionWrite
from motrix_envs.locomotion.wbt.g1.common import MOTION_DIR, G1WbtEnvCfg


@configclass
class RobotOnlyRandomizationCfg:
    enabled: bool = True
    push_enabled: bool = False
    ground_sliding_friction: tuple[float, float] = (0.3, 1.2)
    push_interval_s: tuple[float, float] = (1.0, 3.0)
    push_linear: tuple[float, float, float] = (0.3, 0.3, 0.3)
    push_angular: tuple[float, float, float] = (0.78, 0.78, 0.78)


@configclass(kw_only=True)
class RobotOnlyCarryEnvCfg(G1WbtEnvCfg):
    randomization: RobotOnlyRandomizationCfg = RobotOnlyRandomizationCfg()
    play_training_perturbations: bool = True
    play_ground_friction: float | None = None
    evaluation: bool = False

    def for_play(self):
        cfg = super().for_play()
        cfg.evaluation = True
        cfg.randomization.enabled = cfg.play_training_perturbations
        if not cfg.play_training_perturbations:
            for name in ("motion_ref_ori_b", "base_ang_vel", "dof_pos", "dof_vel"):
                getattr(cfg.observations.policy, name).noise.amplitude = 0.0
        if cfg.play_ground_friction is not None:
            if not np.isfinite(cfg.play_ground_friction) or cfg.play_ground_friction < 0:
                raise ValueError("play_ground_friction must be finite and nonnegative")
            cfg.scene.objs.floor.friction = (cfg.play_ground_friction, *cfg.scene.objs.floor.friction[1:])
        return cfg


@registry.envcfg("g1-29dof-wbt-largebox")
def make_g129dof_wbt_largebox_cfg() -> RobotOnlyCarryEnvCfg:
    """Track the official robot-only carrying motion with the original WBT reward recipe."""
    cfg = RobotOnlyCarryEnvCfg(
        motion_files=(str(MOTION_DIR / "sub3_largebox_003.npz"),)
    )
    cfg.actions.joint_position.safety_clip_fraction = 0.95
    # As in the box ground-DR task, floor material wins contact mixing.
    cfg.scene.objs.floor.priority = 2
    cfg.queries.model["ground_action_friction"] = GeomFrictionQuery(name="floor")
    body = cfg.scene.objs.robot.resolved_base_link_name
    cfg.queries.data["root_lin_vel"] = LinkLinearVelocityQuery(link=body)
    cfg.queries.data["root_ang_vel"] = LinkAngularVelocityQuery(link=body)
    return cfg


@registry.env("g1-29dof-wbt-largebox")
class RobotOnlyCarryEnv(ManagerEnv):
    """Keep per-lane ground draws and apply the box task's velocity-kick pushes."""

    def __init__(self, cfg, num_envs=1, backend=None, seed=None):
        super().__init__(cfg, num_envs=num_envs, backend=backend, seed=seed)
        self._dr_rng = np.random.default_rng(seed)
        self._ground_writes = self.sim.compile_writes({"friction": GeomFrictionWrite(("floor",))}, reset=False)
        friction = self._ground_writes.buffer("friction")
        friction[:, 0] = self.model.others["ground_action_friction"]
        dr = cfg.randomization
        if cfg.evaluation and cfg.play_ground_friction is not None:
            friction[:, 0, 0] = cfg.play_ground_friction
        elif dr.enabled:
            friction[:, 0, 0] = self._dr_rng.uniform(*dr.ground_sliding_friction, size=num_envs)
        body = cfg.scene.objs.robot.resolved_base_link_name
        self._push_writes = self.sim.compile_writes(
            {"linear": BodyLinearVelocityWrite((body,)), "angular": BodyAngularVelocityWrite((body,))},
            reset=False,
        )
        self._push_remaining = self._dr_rng.uniform(*dr.push_interval_s, size=num_envs)
        self._push_linear = np.asarray(dr.push_linear, dtype=np.float32)
        self._push_angular = np.asarray(dr.push_angular, dtype=np.float32)

    def reset(self, env_ids, sim_reset_ids=None):
        super().reset(env_ids, sim_reset_ids)
        selected = env_ids if sim_reset_ids is None else np.union1d(env_ids, sim_reset_ids)
        if selected.size:
            self._ground_writes.execute(selected)
        if env_ids.size:
            self._push_remaining[env_ids] = self._dr_rng.uniform(
                *self.cfg.randomization.push_interval_s, size=env_ids.size
            )

    def physics_step(self):
        if self.cfg.randomization.enabled and self.cfg.randomization.push_enabled:
            self._push_remaining -= self.cfg.ctrl_dt
            selected = np.flatnonzero(self._push_remaining <= 0.0)
            if selected.size:
                self._refresh_sim_reads(selected)
                for key, source, amplitude in (
                    ("linear", "root_lin_vel", self._push_linear),
                    ("angular", "root_ang_vel", self._push_angular),
                ):
                    self._push_writes.buffer(key)[selected, 0] = self.sim_data[source][selected] + self._dr_rng.uniform(
                        -amplitude, amplitude, size=(selected.size, 3)
                    )
                self._push_writes.execute(selected)
                self._push_remaining[selected] = self._dr_rng.uniform(
                    *self.cfg.randomization.push_interval_s, size=selected.size
                )
        super().physics_step()
