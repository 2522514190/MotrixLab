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
from motrix_envs.locomotion.wbt.g1.common import MOTION_DIR


@registry.envcfg("g1-29dof-wbt-largebox-omniretarget-fastsac")
def make_omniretarget_fastsac_cfg() -> OmniRetargetEnvCfg:
    """Build the common FastSAC carrying task configuration."""
    cfg = make_omniretarget_cfg()
    # FastSAC-only trajectory swap: leave the shared PPO recipe untouched.
    motion_path = MOTION_DIR / "sub3_largebox_003_upright_start_end_hold_745.npz"
    if not motion_path.is_file():
        raise FileNotFoundError(
            f"Unilab carrying motion not found: {motion_path}. "
            "Copy the provided NPZ into the G1 motion asset directory first."
        )
    with np.load(motion_path, allow_pickle=False) as motion:
        frames = int(motion["num_frames"])
        fps = int(motion["fps"])
        if frames < 2 or fps != int(round(1.0 / cfg.ctrl_dt)):
            raise ValueError(
                f"Unexpected Unilab motion metadata: frames={frames}, fps={fps}, ctrl_dt={cfg.ctrl_dt}"
            )
    cfg.commands.motion.motion_files = (str(motion_path),)
    cfg.max_episode_seconds = 16.0
    if cfg.max_episode_steps <= frames:
        raise ValueError(
            f"Episode horizon {cfg.max_episode_steps} must exceed {frames} motion frames."
        )
    # Keep both the box curriculum and its delayed error-termination gate
    # aligned to the actual reference, rather than the old 325-frame clip.
    cfg.object_curriculum.reference_frames = frames
    cfg.commands.motion.start_at_timestep_zero_prob = 0.3
    cfg.object_curriculum.from_zero_only = True
    cfg.randomize_initial_episode_progress = False
    cfg.rewards.object_global_ref_position_error_exp.weight = 0.8
    cfg.rewards.object_global_ref_orientation_error_exp.weight = 0.8
    return cfg


def _make_ablation_cfg(
    *, undesired_contact_weight: float, error_termination: bool, adaptive_sampling: bool
) -> OmniRetargetEnvCfg:
    cfg = make_omniretarget_fastsac_cfg()
    cfg.rewards.undesired_contacts.weight = undesired_contact_weight
    cfg.object_curriculum.error_termination_enabled = error_termination
    cfg.commands.motion.adaptive_sampling_enabled = adaptive_sampling
    return cfg


@registry.env("g1-29dof-wbt-largebox-omniretarget-fastsac")
class OmniRetargetFastSacEnv(OmniRetargetEnv):
    """FastSAC task with performance-triggered box-reward curriculum."""

    _CURRICULUM_ALPHA = 1.0 / 256.0
    _WEIGHT_SMOOTHING_ALPHA = 0.1
    # Experiment-specific reward curriculum (existing 0.8 -> 0.2 is the default).
    _BOX_REWARD_INITIAL_WEIGHT = 0.8
    _BOX_REWARD_MIN_WEIGHT = 0.2
    _BOX_REWARD_DECAY_START = 0.8
    _BOX_REWARD_DECAY_END = 0.95
    # None = orientation follows position; a float = independent fixed weight.
    _BOX_ORIENTATION_FIXED_WEIGHT: float | None = None
    # Full-start completion requires sustained reference-aligned lifting, then stable setdown.
    # These checks rely on box height/velocity (not contact-pair sensing).
    _BOX_LIFT_HEIGHT_M = 0.25
    _BOX_LIFT_TRACKING_TOL_M = 0.20
    _BOX_LIFT_HOLD_STEPS = 25  # 0.5 s at 50 Hz
    _BOX_SETDOWN_HEIGHT_M = 0.08
    _BOX_SETDOWN_MAX_SPEED_MPS = 0.20
    _BOX_SETDOWN_HOLD_STEPS = 15  # 0.3 s at 50 Hz

    def __init__(self, cfg: OmniRetargetEnvCfg, num_envs=1, backend=None, seed=None):
        super().__init__(cfg, num_envs=num_envs, backend=backend, seed=seed)
        action = self.action_terms["joint_position"].state
        self._action_space = symmetric_residual_action_space(
            np.stack((action.joint_lower, action.joint_upper)),
            action.default_angles,
            cfg.actions.joint_position.action_scale,
        )
        self._training_iteration = 0
        self._box_weight_indices = None
        self._box_success_bonus = np.zeros(num_envs, dtype=np.float32)
        self._box_success_mask = np.zeros(num_envs, dtype=bool)
        self._box_success_counted = np.zeros(num_envs, dtype=bool)
        self._box_lift_streak = np.zeros(num_envs, dtype=np.int32)
        self._box_lift_verified = np.zeros(num_envs, dtype=bool)
        self._box_setdown_streak = np.zeros(num_envs, dtype=np.int32)
        self._box_setdown_verified = np.zeros(num_envs, dtype=bool)
        self._box_success_total = 0
        self._box_reward_weight = np.float32(self._BOX_REWARD_INITIAL_WEIGHT)
        self._box_curriculum_survival_ratio = np.float32(0.0)
        self._box_curriculum_enabled = False
        self._box_curriculum_trigger_iteration = -1
        self._box_curriculum_state = None
        self._box_curriculum_slot = 0
        self._box_curriculum_ema = None
        self._box_curriculum_count = 0

    def set_training_iteration(self, iteration: int) -> None:
        """Set the provider iteration for curriculum event diagnostics only."""
        self._training_iteration = int(iteration)

    def set_curriculum_state(self, state: np.ndarray, slot: int) -> None:
        """Attach shared collector curriculum state for synchronized weights."""
        self._box_curriculum_state = state
        self._box_curriculum_slot = slot

    def _synchronized_weight(self) -> np.float32:
        if self._box_curriculum_state is None:
            return self._box_reward_weight
        valid = self._box_curriculum_state[:, 1] > 0.0
        if not np.any(valid):
            return self._box_reward_weight
        return np.float32(min(self._box_reward_weight, np.min(self._box_curriculum_state[valid, 2])))

    def _update_curriculum_from_finished(self, env_ids: np.ndarray) -> None:
        frames = self.command_terms["motion"].clip.joint_pos.shape[0]
        ratios = np.minimum(self._elapsed_since_reset[env_ids], frames) / frames
        for ratio in ratios:
            if self._box_curriculum_ema is None:
                self._box_curriculum_ema = float(ratio)
            else:
                self._box_curriculum_ema = (
                    (1.0 - self._CURRICULUM_ALPHA) * self._box_curriculum_ema
                    + self._CURRICULUM_ALPHA * float(ratio)
                )
            self._box_curriculum_count += 1
        if self._box_curriculum_state is not None:
            row = self._box_curriculum_state[self._box_curriculum_slot]
            row[0] = self._box_curriculum_ema
            row[1] = self._box_curriculum_count
            observed = self._box_curriculum_state[:, 0][self._box_curriculum_state[:, 1] > 0.0]
            survival_ratio = float(np.mean(observed)) if observed.size else self._box_curriculum_ema
        else:
            survival_ratio = self._box_curriculum_ema
        progress = np.clip(
            (survival_ratio - self._BOX_REWARD_DECAY_START)
            / (self._BOX_REWARD_DECAY_END - self._BOX_REWARD_DECAY_START),
            0.0,
            1.0,
        )
        target = self._BOX_REWARD_INITIAL_WEIGHT - (
            self._BOX_REWARD_INITIAL_WEIGHT - self._BOX_REWARD_MIN_WEIGHT
        ) * progress
        target = max(target, self._BOX_REWARD_MIN_WEIGHT)
        current = float(self._box_reward_weight)
        smoothed = current + self._WEIGHT_SMOOTHING_ALPHA * (target - current)
        self._box_reward_weight = np.float32(max(self._BOX_REWARD_MIN_WEIGHT, min(current, smoothed)))
        if self._box_curriculum_state is not None:
            row[2] = self._box_reward_weight
            synchronized = self._synchronized_weight()
            if synchronized < self._box_reward_weight:
                self._box_reward_weight = synchronized
        previous_enabled = self._box_curriculum_enabled
        self._box_curriculum_survival_ratio = np.float32(survival_ratio)
        self._box_curriculum_enabled = self._box_reward_weight < self._BOX_REWARD_INITIAL_WEIGHT
        if self._box_curriculum_enabled and not previous_enabled:
            self._box_curriculum_trigger_iteration = self._training_iteration

    def _on_full_start_episodes_finished(self, env_ids):
        self._update_curriculum_from_finished(env_ids)

    def _orientation_reward_weight(self, position_weight: float) -> np.float32:
        if self._BOX_ORIENTATION_FIXED_WEIGHT is not None:
            return np.float32(self._BOX_ORIENTATION_FIXED_WEIGHT)
        return np.float32(position_weight)

    def _apply_box_reward_weights(self, position_weight: float) -> None:
        if self._box_weight_indices is None:
            return
        position_index, orientation_index = self._box_weight_indices
        self._task_program.reward_weights[position_index] = position_weight
        self._task_program.reward_weights[orientation_index] = self._orientation_reward_weight(position_weight)

    def init_state(self):
        state = super().init_state()
        indices = {term.name: term.index for term in self.manager_layout.rewards}
        self._box_weight_indices = tuple(
            indices[name]
            for name in (
                "object_global_ref_position_error_exp",
                "object_global_ref_orientation_error_exp",
            )
        )
        self._apply_box_reward_weights(self._box_reward_weight)
        state.reward_terms["box_success_bonus"] = self._box_success_bonus
        state.metrics["box_success"] = self._box_success_mask
        state.metrics["box_lift_verified"] = self._box_lift_verified
        state.metrics["box_setdown_verified"] = self._box_setdown_verified
        state.metrics["box_reward_weight"] = float(self._box_reward_weight)
        state.metrics["box_position_reward_weight"] = float(self._box_reward_weight)
        state.metrics["box_orientation_reward_weight"] = float(self._orientation_reward_weight(self._box_reward_weight))
        state.metrics["box_curriculum_survival_ratio"] = float(self._box_curriculum_survival_ratio)
        state.metrics["box_curriculum_enabled"] = self._box_curriculum_enabled
        state.metrics["box_curriculum_trigger_iteration"] = float(self._box_curriculum_trigger_iteration)
        state.metrics["box_success_total"] = 0.0
        state.metrics["box_full_start_fraction"] = self._started_from_zero
        return state

    def reset(self, env_ids, sim_reset_ids=None):
        super().reset(env_ids, sim_reset_ids)
        if env_ids.size:
            self._box_success_counted[env_ids] = False
            self._box_lift_streak[env_ids] = 0
            self._box_lift_verified[env_ids] = False
            self._box_setdown_streak[env_ids] = 0
            self._box_setdown_verified[env_ids] = False

    def _track_box_lift_and_setdown(self, motion, object_pos: np.ndarray, object_vel: np.ndarray) -> None:
        """Record a full-start episode's sustained lift followed by stable placement."""
        reference = motion.clip.extensions["object_pos_w"].data
        frame_indices = np.clip(motion.steps[:, 0], 0, len(reference) - 1)
        reference_pos = reference[frame_indices]
        floor_z = float(reference[0, 2])
        tracking_error = np.linalg.norm(object_pos - reference_pos, axis=1)

        # A brief kick or one-frame airborne event does not satisfy the lift stage.
        lifted_now = (
            self._started_from_zero
            & ~self._box_lift_verified
            & (reference_pos[:, 2] >= floor_z + self._BOX_LIFT_HEIGHT_M)
            & (object_pos[:, 2] >= floor_z + self._BOX_LIFT_HEIGHT_M)
            & (tracking_error <= self._BOX_LIFT_TRACKING_TOL_M)
        )
        self._box_lift_streak[:] = np.where(lifted_now, self._box_lift_streak + 1, 0)
        self._box_lift_verified |= self._box_lift_streak >= self._BOX_LIFT_HOLD_STEPS

        placed_now = (
            self._started_from_zero
            & self._box_lift_verified
            & ~self._box_setdown_verified
            & (reference_pos[:, 2] <= floor_z + self._BOX_SETDOWN_HEIGHT_M)
            & (abs(object_pos[:, 2] - floor_z) <= self._BOX_SETDOWN_HEIGHT_M)
            & (tracking_error <= self._BOX_LIFT_TRACKING_TOL_M)
            & (np.linalg.norm(object_vel, axis=1) <= self._BOX_SETDOWN_MAX_SPEED_MPS)
        )
        self._box_setdown_streak[:] = np.where(placed_now, self._box_setdown_streak + 1, 0)
        self._box_setdown_verified |= self._box_setdown_streak >= self._BOX_SETDOWN_HOLD_STEPS

    def compute_transition(self, state):
        weight = self._synchronized_weight()
        self._box_reward_weight = weight
        self._apply_box_reward_weights(weight)
        state = super().compute_transition(state)
        self._box_success_bonus.fill(0.0)
        self._box_success_mask.fill(False)
        motion = self.command_terms["motion"]
        object_pos = self._compiled_manager_program.read_plan.sim_data.view("object_pos")
        object_vel = self._compiled_manager_program.read_plan.sim_data.view("object_lin_vel")
        self._track_box_lift_and_setdown(motion, object_pos, object_vel)
        object_quat = self._compiled_manager_program.read_plan.sim_data.view("object_quat")
        reference_pos = motion.clip.extensions["object_pos_w"].data[-1]
        reference_quat = motion.clip.extensions["object_quat_w"].data[-1]
        pos_error = np.linalg.norm(object_pos - reference_pos, axis=1)
        quat_dot = np.sum(object_quat * reference_quat, axis=1)
        quat_den = np.maximum(np.linalg.norm(object_quat, axis=1) * np.linalg.norm(reference_quat), 1e-8)
        ori_error = 2.0 * np.arccos(np.clip(np.abs(quat_dot) / quat_den, 0.0, 1.0))
        reached_end = self._started_from_zero & motion.clip_ended[:, 0] & ~self._box_success_counted
        success = (
            reached_end
            & self._box_lift_verified
            & self._box_setdown_verified
            & (pos_error < 0.15)
            & (ori_error < 0.30)
            & ~state.terminated
        )
        self._box_success_bonus[success] = 10.0
        self._box_success_mask[success] = True
        self._box_success_counted[reached_end] = True
        self._box_success_total += int(np.count_nonzero(success))
        state.reward[:] += self._box_success_bonus
        state.terminated[reached_end] = True
        state.metrics["box_reward_weight"] = float(weight)
        state.metrics["box_position_reward_weight"] = float(weight)
        state.metrics["box_orientation_reward_weight"] = float(self._orientation_reward_weight(weight))
        state.metrics["box_curriculum_survival_ratio"] = float(self._box_curriculum_survival_ratio)
        state.metrics["box_curriculum_enabled"] = self._box_curriculum_enabled
        state.metrics["box_curriculum_trigger_iteration"] = float(self._box_curriculum_trigger_iteration)
        state.metrics["box_success_total"] = float(self._box_success_total)
        return state


@registry.envcfg("g1-29dof-wbt-largebox-omniretarget-fastsac-a")
def make_a_cfg() -> OmniRetargetEnvCfg:
    return _make_ablation_cfg(undesired_contact_weight=-0.1, error_termination=True, adaptive_sampling=True)


@registry.envcfg("g1-29dof-wbt-largebox-omniretarget-fastsac-b")
def make_b_cfg() -> OmniRetargetEnvCfg:
    return _make_ablation_cfg(undesired_contact_weight=0.0, error_termination=True, adaptive_sampling=True)


@registry.envcfg("g1-29dof-wbt-largebox-omniretarget-fastsac-c")
def make_c_cfg() -> OmniRetargetEnvCfg:
    return _make_ablation_cfg(undesired_contact_weight=-0.1, error_termination=False, adaptive_sampling=True)


@registry.envcfg("g1-29dof-wbt-largebox-omniretarget-fastsac-d")
def make_d_cfg() -> OmniRetargetEnvCfg:
    return _make_ablation_cfg(undesired_contact_weight=-0.1, error_termination=True, adaptive_sampling=False)


@registry.envcfg("g1-29dof-wbt-largebox-omniretarget-fastsac-e")
def make_e_cfg() -> OmniRetargetEnvCfg:
    return _make_ablation_cfg(undesired_contact_weight=0.0, error_termination=False, adaptive_sampling=True)


@registry.envcfg("g1-29dof-wbt-largebox-omniretarget-fastsac-f")
def make_f_cfg() -> OmniRetargetEnvCfg:
    return _make_ablation_cfg(undesired_contact_weight=0.0, error_termination=True, adaptive_sampling=False)


@registry.envcfg("g1-29dof-wbt-largebox-omniretarget-fastsac-g")
def make_g_cfg() -> OmniRetargetEnvCfg:
    return _make_ablation_cfg(undesired_contact_weight=-0.1, error_termination=False, adaptive_sampling=False)


@registry.envcfg("g1-29dof-wbt-largebox-omniretarget-fastsac-h")
def make_h_cfg() -> OmniRetargetEnvCfg:
    return _make_ablation_cfg(undesired_contact_weight=0.0, error_termination=False, adaptive_sampling=False)


for _suffix in "abcdefgh":
    registry.env(f"g1-29dof-wbt-largebox-omniretarget-fastsac-{_suffix}")(OmniRetargetFastSacEnv)


# Three reward ablations use identical FastSAC / contact / termination settings.
# Fix the contact penalty to B (0.0); only the named box-reward dimension varies.
def _make_box_reward_ablation_cfg() -> OmniRetargetEnvCfg:
    cfg = _make_ablation_cfg(undesired_contact_weight=0.0, error_termination=True, adaptive_sampling=True)
    cfg.rewards.object_global_ref_position_error_exp.weight = 0.5
    cfg.rewards.object_global_ref_orientation_error_exp.weight = 0.5
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-omniretarget-fastsac-reward05-80")
def make_box_reward05_80_cfg() -> OmniRetargetEnvCfg:
    """Ablation 1: both box rewards 0.5 -> 0.2 over 80%-95% survival."""
    return _make_box_reward_ablation_cfg()


@registry.envcfg("g1-29dof-wbt-largebox-omniretarget-fastsac-reward05-50")
def make_box_reward05_50_cfg() -> OmniRetargetEnvCfg:
    """Ablation 2: both box rewards 0.5 -> 0.2 over 50%-95% survival."""
    return _make_box_reward_ablation_cfg()


@registry.envcfg("g1-29dof-wbt-largebox-omniretarget-fastsac-reward05-50-orifixed")
def make_box_reward05_50_orifixed_cfg() -> OmniRetargetEnvCfg:
    """Ablation 3: position 0.5 -> 0.2 over 50%-95%, orientation fixed at 0.5."""
    return _make_box_reward_ablation_cfg()


class OmniRetargetReward05_80Env(OmniRetargetFastSacEnv):
    _BOX_REWARD_INITIAL_WEIGHT = 0.5
    _BOX_REWARD_DECAY_START = 0.8


class OmniRetargetReward05_50Env(OmniRetargetFastSacEnv):
    _BOX_REWARD_INITIAL_WEIGHT = 0.5
    _BOX_REWARD_DECAY_START = 0.5


class OmniRetargetReward05_50OriFixedEnv(OmniRetargetReward05_50Env):
    _BOX_ORIENTATION_FIXED_WEIGHT = 0.5


registry.env("g1-29dof-wbt-largebox-omniretarget-fastsac-reward05-80")(OmniRetargetReward05_80Env)
registry.env("g1-29dof-wbt-largebox-omniretarget-fastsac-reward05-50")(OmniRetargetReward05_50Env)
registry.env("g1-29dof-wbt-largebox-omniretarget-fastsac-reward05-50-orifixed")(
    OmniRetargetReward05_50OriFixedEnv
)
