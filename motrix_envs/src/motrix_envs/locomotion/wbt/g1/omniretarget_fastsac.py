# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0

"""OmniRetarget task with Holosoma's finite FastSAC policy-action bounds."""

import numpy as np

from motrix_env_core import registry
from motrix_env_core.mdp.action_space import symmetric_residual_action_space
from motrix_envs.locomotion.wbt.g1.omniretarget import (
    OmniRetargetEnv,
    OmniRetargetEnvCfg,
    make_omniretarget_cfg,
)


@registry.envcfg("g1-29dof-wbt-largebox-omniretarget-fastsac")
def make_omniretarget_fastsac_cfg() -> OmniRetargetEnvCfg:
    """Reuse the PPO task configuration, including its unbounded PD setpoints.

    zh_CN: 复用 PPO 搬箱配置，仅由任务运行时提供 FastSAC 的有限策略动作边界。
    """
    return make_omniretarget_cfg()


@registry.env("g1-29dof-wbt-largebox-omniretarget-fastsac")
class OmniRetargetFastSacEnv(OmniRetargetEnv):
    """Bound policy output without changing the shared calibrated PD mapping.

    Holosoma ``FastSACEnvWrapper._compute_action_boundaries`` uses the largest
    distance from the default pose to a mechanical joint limit, divided by the
    scalar robot ``control.action_scale``. Preserve that scalar convention;
    the inherited action term still applies the per-joint effort/kp scales.
    """

    def __init__(self, cfg: OmniRetargetEnvCfg, num_envs=1, backend=None, seed=None):
        super().__init__(cfg, num_envs=num_envs, backend=backend, seed=seed)
        action = self.action_terms["joint_position"].state
        self._action_space = symmetric_residual_action_space(
            np.stack((action.joint_lower, action.joint_upper)),
            action.default_angles,
            cfg.actions.joint_position.action_scale,
        )

        # FastSAC-only curriculum; the reward kernel reads mutable weights.
        self._box_decay_steps = 80000
        self._box_decay_elapsed = 0
        self._box_weight_indices = None
        self._box_success_bonus = np.zeros(num_envs, dtype=np.float32)
        self._box_success_counted = np.zeros(num_envs, dtype=bool)
        self._box_stable_steps = np.zeros(num_envs, dtype=np.int32)
        self._box_was_lifted = np.zeros(num_envs, dtype=bool)
        self._box_was_carried = np.zeros(num_envs, dtype=bool)
        self._box_success_total = 0

    def init_state(self):
        state = super().init_state()
        indices = {term.name: term.index for term in self.manager_layout.rewards}
        names = (
            "object_global_ref_position_error_exp",
            "object_global_ref_orientation_error_exp",
        )
        self._box_weight_indices = tuple(indices[name] for name in names)
        self._task_program.reward_weights[list(self._box_weight_indices)] = 0.5
        # Observable through FastSAC reward-term and metric logs.
        state.reward_terms["box_success_bonus"] = self._box_success_bonus
        state.metrics["box_success"] = self._box_success_bonus / np.float32(5.0)
        state.metrics["box_reward_weight"] = 0.5
        state.metrics["box_success_total"] = 0.0
        state.metrics["box_full_start_fraction"] = self._started_from_zero.astype(np.float32)
        return state

    def reset(self, env_ids, sim_reset_ids=None):
        super().reset(env_ids, sim_reset_ids)
        if env_ids.size:
            self._box_success_counted[env_ids] = False
            self._box_stable_steps[env_ids] = 0
            self._box_was_lifted[env_ids] = False
            self._box_was_carried[env_ids] = False

    def compute_transition(self, state):
        state = super().compute_transition(state)
        self._box_success_bonus.fill(0.0)
        motion = self.command_terms["motion"]
        object_pos = self.sim_data["object_pos"]
        object_quat = self.sim_data["object_quat"]
        reference_pos = motion.clip.extensions["object_pos_w"].data
        reference_quat = motion.clip.extensions["object_quat_w"].data
        self._box_was_lifted |= (
            self._started_from_zero & (object_pos[:, 2] - reference_pos[0, 2] > 0.25)
        )
        self._box_was_carried |= (
            self._started_from_zero
            & (np.linalg.norm(object_pos[:, :2] - reference_pos[0, :2], axis=1) > 0.50)
        )
        pos_error = np.linalg.norm(object_pos - reference_pos[-1], axis=1)
        quat_dot = np.sum(object_quat * reference_quat[-1], axis=1)
        quat_den = np.maximum(
            np.linalg.norm(object_quat, axis=1) * np.linalg.norm(reference_quat[-1]), 1e-8
        )
        ori_error = 2.0 * np.arccos(np.clip(np.abs(quat_dot) / quat_den, 0.0, 1.0))
        lin_speed = np.linalg.norm(self.sim_data["object_lin_vel"], axis=1)
        ang_speed = np.linalg.norm(self.sim_data["object_ang_vel"], axis=1)
        # Ten consecutive 50-Hz frames verify 0.2 s of stable placement.
        in_final_window = (
            self._started_from_zero
            & (self._elapsed_since_reset >= reference_pos.shape[0] - 10)
            & (self._elapsed_since_reset <= reference_pos.shape[0])
        )
        stable = (
            in_final_window
            & (pos_error < 0.15)
            & (ori_error < 0.30)
            & (lin_speed < 0.15)
            & (ang_speed < 0.30)
            & ~state.terminated
        )
        self._box_stable_steps[~stable] = 0
        self._box_stable_steps[stable] += 1
        reached_end = (
            self._started_from_zero
            & motion.clip_ended[:, 0]
            & (self._elapsed_since_reset >= reference_pos.shape[0])
            & ~self._box_success_counted
        )
        success = (
            reached_end & self._box_was_lifted & self._box_was_carried
            & (self._box_stable_steps >= 10) & ~state.terminated
        )
        self._box_success_bonus[success] = 5.0
        self._box_success_counted[reached_end] = True
        self._box_success_total += int(np.count_nonzero(success))
        state.reward[:] += self._box_success_bonus
        # Avoid a teleport discontinuity within the same replay-buffer episode.
        state.terminated[reached_end] = True
        if self._gate.enabled[0] and not self.cfg.evaluation:
            self._box_decay_elapsed = min(self._box_decay_elapsed + 1, self._box_decay_steps)
        fraction = self._box_decay_elapsed / self._box_decay_steps
        weight = np.float32(0.5 + (0.2 - 0.5) * fraction)
        if self._box_weight_indices is not None:
            self._task_program.reward_weights[list(self._box_weight_indices)] = weight
        state.metrics["box_reward_weight"] = float(weight)
        state.metrics["box_success_total"] = float(self._box_success_total)
        return state
