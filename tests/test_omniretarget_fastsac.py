# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-side FastSAC bounds must preserve the shared Manager control contract."""

import numpy as np
import pytest
import torch
from motrix_rl_builtin.fastsac.networks import Actor
from motrix_rl_builtin.fastsac.wrap_np import FastSacNpEnvWrap

import motrix_envs  # noqa: F401 registers the task before resolving it
from motrix_env_core import registry


@pytest.fixture(scope="module")
def wrapped_env():
    spec = registry.resolve("g1-29dof-wbt-largebox-omniretarget-fastsac")
    env = spec.make(num_envs=2, seed=23)
    wrapped = FastSacNpEnvWrap(env, torch.device("cpu"))
    wrapped.reset()
    yield wrapped
    wrapped.close()


@pytest.mark.numba
def test_finite_holosoma_bounds_preserve_unbounded_pd_and_calibrated_targets(wrapped_env):
    env = wrapped_env.env
    term = env.action_terms["joint_position"]
    state = term.state
    bounds = env.action_space.high
    expected = (
        np.maximum(np.abs(state.joint_lower - state.default_angles), np.abs(state.joint_upper - state.default_angles))
        / env.cfg.actions.joint_position.action_scale
    )
    np.testing.assert_allclose(bounds, expected)
    np.testing.assert_array_equal(env.action_space.low, -bounds)
    assert np.all(np.isfinite(bounds)) and np.all(bounds > 0)
    # Policy support must not turn into actuator position-target clipping.
    assert np.all(np.isfinite(term.action_space.low))
    assert np.all(np.isfinite(term.action_space.high))
    actions = np.broadcast_to(bounds * 0.2, (env.num_envs, bounds.size)).copy()
    targets = term.process(actions)
    expected = state.default_angles + state.bias + actions * state.action_scales
    midpoint = (state.joint_lower + state.joint_upper) * 0.5
    half_range = (state.joint_upper - state.joint_lower) * (0.5 * env.cfg.actions.joint_position.safety_clip_fraction)
    np.testing.assert_allclose(targets, np.clip(expected, midpoint - half_range, midpoint + half_range), atol=1e-6)
    np.testing.assert_array_equal(state.current(), actions)


@pytest.mark.numba
def test_fastsac_actor_and_wrapper_use_privileged_observations_and_finite_actions(wrapped_env):
    policy, critic = wrapped_env.reset()
    env = wrapped_env.env
    assert policy.shape == (env.num_envs, *env.policy_observation_space.shape)
    assert critic.shape == (env.num_envs, *env.value_observation_space.shape)
    assert critic.shape[1] > policy.shape[1]
    low, high = wrapped_env.action_low, wrapped_env.action_high
    actor = Actor(
        n_obs=policy.shape[1],
        n_act=low.numel(),
        hidden_dim=32,
        log_std_max=0,
        log_std_min=-5,
        action_scale=(high - low) / 2,
        action_bias=(high + low) / 2,
    )
    actions, log_probs = actor.get_actions_and_log_probs(policy)
    assert torch.isfinite(actions).all() and torch.isfinite(log_probs).all()
    assert ((actions >= low) & (actions <= high)).all()
    result = wrapped_env.step(actions)
    policy, critic, rewards, terminated, truncated = (
        result.obs,
        result.critic_obs,
        result.reward,
        result.terminated,
        result.truncated,
    )
    assert all(torch.isfinite(value).all() for value in (policy, critic, rewards))
    assert terminated.dtype == truncated.dtype == torch.bool
    np.testing.assert_array_equal(terminated.numpy(), env.state.terminated)
    np.testing.assert_array_equal(truncated.numpy(), env.state.truncated)


@pytest.mark.numba
def test_fastsac_wrapper_preserves_timeout_without_turning_it_into_failure(wrapped_env):
    env = wrapped_env.env
    wrapped_env.reset()
    # Force a timeout while the physical state is still the reset reference.
    env.state.episode_steps[:] = env.cfg.max_episode_steps - 1
    env._gate.enabled[0] = False
    result = wrapped_env.step(torch.zeros((env.num_envs, env.action_space.shape[0])))
    terminated, truncated = result.terminated, result.truncated
    assert truncated.all()
    assert not terminated.any()


@pytest.mark.numba
def test_carry_curriculum_preserves_adaptive_sampling_and_updates_compiled_weights():
    spec = registry.resolve("g1-29dof-wbt-largebox-omniretarget-fastsac")
    env = spec.make(num_envs=2, seed=17)
    try:
        env.init_state()
        assert env.cfg.commands.motion.adaptive_sampling_enabled
        assert env.cfg.commands.motion.start_at_timestep_zero_prob == pytest.approx(0.3)
        assert env.cfg.object_curriculum.from_zero_only
        assert not env.cfg.randomize_initial_episode_progress
        assert env.cfg.max_episode_steps == 800
        assert env.cfg.rewards.object_global_ref_position_error_exp.weight == pytest.approx(0.8)
        assert env.cfg.rewards.object_global_ref_orientation_error_exp.weight == pytest.approx(0.8)
        assert env.cfg.object_curriculum.reference_frames == 745
        assert env.command_terms["motion"].clip.joint_pos.shape[0] == 745
        assert env.cfg.commands.motion.motion_files[0].endswith("sub3_largebox_003_upright_start_end_hold_745.npz")
        assert not env._gate.enabled[0]
        indices = env._box_weight_indices
        np.testing.assert_allclose(env._task_program.reward_weights[list(indices)], 0.8)
        assert "box_success_bonus" in env.state.reward_terms
        assert "box_success" in env.state.metrics
        assert "box_lift_verified" in env.state.metrics
        assert "box_setdown_verified" in env.state.metrics

        # Curriculum is performance-triggered and independent of the error gate.
        env._gate.enabled[0] = False
        env._elapsed_since_reset[:] = 100
        env._on_full_start_episodes_finished(np.array([0, 1], dtype=np.int64))
        assert env._box_reward_weight == pytest.approx(0.8)
        for _ in range(256):
            env._elapsed_since_reset[:] = env.cfg.object_curriculum.reference_frames
            env._on_full_start_episodes_finished(np.array([0, 1], dtype=np.int64))
        assert env._box_curriculum_enabled
        assert env._box_reward_weight < 0.8
        assert env._box_reward_weight >= 0.2
        # Re-evaluate with the new weights; compare the compiled kernel's
        # weighted term against the raw term, not just the cfg value.
        env._task_program.reward_weights[list(indices)] = env._box_reward_weight
        env._refresh_sim_reads()
        env._execute_evaluate_kernel(env._kernel_inputs)
        for index in indices:
            np.testing.assert_allclose(
                env._kernel_buffers[1][:, index],
                env._kernel_buffers[0][:, index] * float(env._box_reward_weight) * env.cfg.ctrl_dt,
                atol=1e-6,
            )
    finally:
        del env


@pytest.mark.numba
def test_full_start_sample_and_live_reward_metrics():
    spec = registry.resolve("g1-29dof-wbt-largebox-omniretarget-fastsac")
    cfg = registry.make_env_config("g1-29dof-wbt-largebox-omniretarget-fastsac")
    cfg.commands.motion.start_at_timestep_zero_prob = 1.0
    from motrix_envs.locomotion.wbt.g1.omniretarget_fastsac import OmniRetargetFastSacEnv
    env = OmniRetargetFastSacEnv(cfg, num_envs=2, seed=29)
    try:
        env.init_state()
        assert env._started_from_zero.all()
        assert np.shares_memory(env.state.reward_terms["box_success_bonus"], env._box_success_bonus)
        assert np.shares_memory(env.state.metrics["box_success"], env._box_success_mask)
    finally:
        del env

@pytest.mark.numba
def test_carry_ablation_configs_control_sampling_and_error_gate():
    expected = {
        "a": (-0.1, True, True), "b": (0.0, True, True),
        "c": (-0.1, False, True), "d": (-0.1, True, False),
        "e": (0.0, False, True), "f": (0.0, True, False),
        "g": (-0.1, False, False), "h": (0.0, False, False),
    }
    for suffix, (contact, error_term, adaptive) in expected.items():
        spec = registry.resolve(f"g1-29dof-wbt-largebox-omniretarget-fastsac-{suffix}")
        env = spec.make(num_envs=1, seed=7)
        try:
            assert env.cfg.rewards.undesired_contacts.weight == pytest.approx(contact)
            assert env.cfg.object_curriculum.error_termination_enabled is error_term
            assert env.cfg.commands.motion.adaptive_sampling_enabled is adaptive
            assert env.cfg.commands.motion.start_at_timestep_zero_prob == pytest.approx(0.3)
            if not error_term:
                assert not bool(env._gate.enabled[0])
        finally:
            del env



@pytest.mark.numba
@pytest.mark.parametrize(
    ("suffix", "decay_start", "fixed_orientation", "should_decay_at_65"),
    [
        ("reward05-80", 0.8, False, False),
        ("reward05-50", 0.5, False, True),
        ("reward05-50-orifixed", 0.5, True, True),
    ],
)
def test_box_reward_ablation_variants_are_isolated_and_write_independent_weights(
    suffix, decay_start, fixed_orientation, should_decay_at_65
):
    spec = registry.resolve(f"g1-29dof-wbt-largebox-omniretarget-fastsac-{suffix}")
    env = spec.make(num_envs=2, seed=41)
    try:
        env.init_state()
        cfg = env.cfg
        assert cfg.commands.motion.motion_files[0].endswith("sub3_largebox_003_upright_start_end_hold_745.npz")
        assert cfg.object_curriculum.reference_frames == 745
        assert cfg.commands.motion.adaptive_sampling_enabled
        assert cfg.commands.motion.start_at_timestep_zero_prob == pytest.approx(0.3)
        assert cfg.object_curriculum.from_zero_only
        assert cfg.object_curriculum.error_termination_enabled
        assert cfg.rewards.undesired_contacts.weight == pytest.approx(0.0)
        assert cfg.rewards.object_global_ref_position_error_exp.weight == pytest.approx(0.5)
        assert cfg.rewards.object_global_ref_orientation_error_exp.weight == pytest.approx(0.5)
        assert not env._gate.enabled[0]
        assert env._BOX_REWARD_DECAY_START == pytest.approx(decay_start)
        assert env._BOX_REWARD_DECAY_END == pytest.approx(0.95)
        assert env._BOX_REWARD_INITIAL_WEIGHT == pytest.approx(0.5)
        assert env._BOX_REWARD_MIN_WEIGHT == pytest.approx(0.2)
        assert (env._BOX_ORIENTATION_FIXED_WEIGHT is not None) is fixed_orientation

        position_idx, orientation_idx = env._box_weight_indices
        weights = env._task_program.reward_weights
        np.testing.assert_allclose(weights[[position_idx, orientation_idx]], 0.5)
        assert env.state.metrics["box_position_reward_weight"] == pytest.approx(0.5)
        assert env.state.metrics["box_orientation_reward_weight"] == pytest.approx(0.5)

        # Hold the survival ratio near 65% to isolate the onset change:
        # 80%-start must not decay, 50%-start must decay.
        env._box_curriculum_ema = 0.65
        env._elapsed_since_reset[:] = int(round(0.65 * 745))
        for _ in range(12):
            env._on_full_start_episodes_finished(np.array([0, 1], dtype=np.int64))
        assert bool(env._box_reward_weight < 0.5) == should_decay_at_65

        # Both dynamic variants eventually decay, but the third variant's
        # orientation stays at 0.5 in the *compiled* reward weights.
        env._box_curriculum_ema = 0.90
        env._elapsed_since_reset[:] = int(round(0.90 * 745))
        for _ in range(12):
            env._on_full_start_episodes_finished(np.array([0, 1], dtype=np.int64))
        assert 0.2 <= env._box_reward_weight < 0.5
        env._apply_box_reward_weights(env._box_reward_weight)
        assert weights[position_idx] == pytest.approx(env._box_reward_weight)
        expected_orientation = 0.5 if fixed_orientation else env._box_reward_weight
        assert weights[orientation_idx] == pytest.approx(expected_orientation)
    finally:
        del env



@pytest.mark.numba
@pytest.mark.parametrize(
    ("suffix", "initial"),
    [
        ("linear08-100k", 0.8),
        ("linear05-100k", 0.5),
        ("linear10-100k", 1.0),
    ],
)
def test_linear_iteration_box_rewards_are_exact_and_independent_of_survival(suffix, initial):
    name = f"g1-29dof-wbt-largebox-omniretarget-fastsac-{suffix}"
    env = registry.resolve(name).make(num_envs=2, seed=43)
    try:
        env.init_state()
        cfg = env.cfg
        assert cfg.commands.motion.motion_files[0].endswith("sub3_largebox_003_upright_start_end_hold_745.npz")
        assert cfg.object_curriculum.reference_frames == 745
        assert cfg.object_curriculum.from_zero_only
        assert cfg.object_curriculum.error_termination_enabled
        assert cfg.object_curriculum.enable_at_horizon_fraction == pytest.approx(0.8)
        assert cfg.commands.motion.adaptive_sampling_enabled
        assert cfg.commands.motion.start_at_timestep_zero_prob == pytest.approx(0.3)
        assert cfg.rewards.undesired_contacts.weight == pytest.approx(0.0)
        assert cfg.rewards.object_global_ref_position_error_exp.weight == pytest.approx(initial)
        assert cfg.rewards.object_global_ref_orientation_error_exp.weight == pytest.approx(initial)
        assert env._BOX_REWARD_TOTAL_ITERS == 100_000
        assert env._BOX_REWARD_MIN_WEIGHT == pytest.approx(0.2)
        assert env._BOX_ORIENTATION_FIXED_WEIGHT is None
        assert not env._gate.enabled[0]

        position_idx, orientation_idx = env._box_weight_indices
        weights = env._task_program.reward_weights
        for iteration in (0, 25_000, 50_000, 75_000, 100_000, 120_000):
            env.set_training_iteration(iteration)
            fraction = min(iteration / 100_000, 1.0)
            expected = initial + (0.2 - initial) * fraction
            assert env._box_reward_weight == pytest.approx(expected, abs=1e-6)
            env._apply_box_reward_weights(env._synchronized_weight())
            np.testing.assert_allclose(weights[[position_idx, orientation_idx]], expected, atol=1e-6)

        # The survival ratio may jump, but the weight must *only* reflect
        # collector iterations. Likewise, it must not enable the object gate.
        env.set_training_iteration(50_000)
        before = float(env._box_reward_weight)
        env._elapsed_since_reset[:] = 745
        for _ in range(24):
            env._on_full_start_episodes_finished(np.array([0, 1], dtype=np.int64))
        assert float(env._box_reward_weight) == pytest.approx(before)
        assert env._box_curriculum_survival_ratio >= 0.9
        assert not env._gate.enabled[0]
        assert env.state.metrics["box_lift_verified"].shape == (env.num_envs,)
        assert env.state.metrics["box_setdown_verified"].shape == (env.num_envs,)
    finally:
        del env



@pytest.mark.numba
def test_reward05_80_fixed_30k_object_termination_gate_and_thresholds():
    """The ONLY change from reward05-80 is a forced gate at 30k and angle=0.8."""
    from unittest.mock import patch

    from motrix_envs.locomotion.wbt.g1.omniretarget_fastsac import (
        OmniRetargetReward05_80Env,
        OmniRetargetReward05_80Term30kEnv,
    )

    name = "g1-29dof-wbt-largebox-omniretarget-fastsac-reward05-80-term30k"
    env = registry.resolve(name).make(num_envs=2, seed=47)
    try:
        env.init_state()
        cfg = env.cfg
        assert isinstance(env, OmniRetargetReward05_80Term30kEnv)
        assert cfg.rewards.object_global_ref_position_error_exp.weight == pytest.approx(0.5)
        assert cfg.rewards.object_global_ref_orientation_error_exp.weight == pytest.approx(0.5)
        assert env._BOX_REWARD_DECAY_START == pytest.approx(0.8)
        assert env._BOX_REWARD_DECAY_END == pytest.approx(0.95)
        assert env._BOX_REWARD_MIN_WEIGHT == pytest.approx(0.2)
        assert cfg.rewards.undesired_contacts.weight == pytest.approx(0.0)
        assert cfg.object_curriculum.error_termination_enabled
        assert cfg.commands.motion.adaptive_sampling_enabled
        assert cfg.commands.motion.start_at_timestep_zero_prob == pytest.approx(0.3)
        assert cfg.object_curriculum.from_zero_only
        assert cfg.object_curriculum.reference_frames == 745
        assert cfg.terminations.object_tracking.position == pytest.approx(1.0)
        assert cfg.terminations.object_tracking.angle == pytest.approx(0.8)
        assert env._BOX_TERMINATION_ENABLE_ITERATION == 30_000
        assert not bool(env._gate.enabled[0])

        env.set_training_iteration(29_999)
        assert not bool(env._gate.enabled[0])
        env._refresh_sim_reads()
        env._execute_evaluate_kernel(env._kernel_inputs)
        np.testing.assert_allclose(env.state.metrics["object_termination_enabled"], 0.0)

        env.set_training_iteration(30_000)
        assert bool(env._gate.enabled[0])
        env._refresh_sim_reads()
        env._execute_evaluate_kernel(env._kernel_inputs)
        # Check the actual compiled termination kernel, not just the Python
        # gate flag; this catches the earlier reported never-enabled behavior.
        np.testing.assert_allclose(env.state.metrics["object_termination_enabled"], 1.0)
        env.set_training_iteration(50_000)
        assert bool(env._gate.enabled[0])

        # Guard against the parent reset() enabling the survival-based gate
        # prematurely: subclass must override it at every reset.
        env.set_training_iteration(29_999)
        with patch.object(
            OmniRetargetReward05_80Env,
            "reset",
            autospec=True,
            side_effect=lambda obj, env_ids, sim_reset_ids=None: obj._gate.enabled.__setitem__(0, True),
        ):
            env.reset(np.array([0], dtype=np.int64))
        assert not bool(env._gate.enabled[0])

        # The fixed gate remains on through later resets regardless of EMA.
        env.set_training_iteration(30_000)
        with patch.object(
            OmniRetargetReward05_80Env,
            "reset",
            autospec=True,
            side_effect=lambda obj, env_ids, sim_reset_ids=None: obj._gate.enabled.__setitem__(0, False),
        ):
            env.reset(np.array([0], dtype=np.int64))
        assert bool(env._gate.enabled[0])
    finally:
        del env



@pytest.mark.numba
def test_reward05_50_bimanual_contact_isolated_from_existing_ablation():
    name = "g1-29dof-wbt-largebox-omniretarget-fastsac-reward05-50-bimanual"
    env = registry.resolve(name).make(num_envs=2, seed=59)
    try:
        env.init_state()
        cfg = env.cfg
        assert cfg.commands.motion.motion_files[0].endswith("sub3_largebox_003_upright_start_end_hold_745.npz")
        assert cfg.commands.motion.adaptive_sampling_enabled
        assert cfg.object_curriculum.from_zero_only
        assert cfg.object_curriculum.error_termination_enabled
        assert cfg.rewards.undesired_contacts.weight == pytest.approx(0.0)
        assert cfg.rewards.object_global_ref_position_error_exp.weight == pytest.approx(0.5)
        assert cfg.rewards.object_global_ref_orientation_error_exp.weight == pytest.approx(0.5)
        assert env._BOX_REWARD_INITIAL_WEIGHT == pytest.approx(0.5)
        assert env._BOX_REWARD_MIN_WEIGHT == pytest.approx(0.2)
        assert env._BOX_REWARD_DECAY_START == pytest.approx(0.5)
        assert env._BOX_REWARD_DECAY_END == pytest.approx(0.95)
        assert env._BOX_ORIENTATION_FIXED_WEIGHT is None
        assert cfg.actions.joint_position.clip_target is False

        # Confirm the sim backend can compile two named geom collision pairs.
        contact = env._compiled_manager_program.read_plan.sim_data.view("hand_box_contacts")
        assert contact.shape == (env.num_envs, 2)
        assert "box_bimanual_contact_reward" in env.state.reward_terms
        assert "box_bimanual_contact" in env.state.metrics
        assert "box_bimanual_phase" in env.state.metrics

        # Finite FastSAC actor actions remain bounded, but no *extra* PD
        # target clamp is allowed in this single experimental environment.
        term = env.action_terms["joint_position"]
        bounds = env.action_space.high
        actions = np.broadcast_to(bounds * 0.95, (env.num_envs, bounds.size)).copy()
        targets = term.process(actions)
        expected = term.state.default_angles + term.state.bias + actions * term.state.action_scales
        np.testing.assert_allclose(targets, expected, rtol=1e-5, atol=1e-6)

        # AND semantics: one hand alone yields zero, both yield +weight*dt.
        env.state.reward[:] = 0.0
        both_and_single = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        env._apply_bimanual_contact_reward(
            env.state, both_and_single, np.array([True, True], dtype=bool)
        )
        np.testing.assert_array_equal(env._bimanual_contact_mask, [True, False])
        np.testing.assert_allclose(
            env._bimanual_contact_reward,
            [env._BIMANUAL_CONTACT_WEIGHT * cfg.ctrl_dt, 0.0],
        )

        # Contact outside the reference lifting/carrying phase is not rewarded.
        env.state.reward[:] = 0.0
        env._apply_bimanual_contact_reward(
            env.state, np.ones((2, 2), dtype=np.float32), np.array([False, True], dtype=bool)
        )
        np.testing.assert_allclose(
            env._bimanual_contact_reward,
            [0.0, env._BIMANUAL_CONTACT_WEIGHT * cfg.ctrl_dt],
        )
    finally:
        del env


@pytest.mark.numba
def test_reward05_50_original_retains_calibrated_target_clipping():
    name = "g1-29dof-wbt-largebox-omniretarget-fastsac-reward05-50"
    env = registry.resolve(name).make(num_envs=1, seed=61)
    try:
        env.init_state()
        assert env.cfg.actions.joint_position.clip_target is True
        assert env.cfg.actions.joint_position.safety_clip_fraction == pytest.approx(0.95)
    finally:
        del env


@pytest.mark.numba
def test_unilab_full_start_bonus_is_ten_once_and_not_dt_scaled():
    """Test host-side success injection while keeping unrelated termination checks isolated."""
    from unittest.mock import patch

    from motrix_env_core.sim.write import BodyPositionWrite, BodyRotationWrite
    from motrix_envs.locomotion.wbt.g1.omniretarget import OmniRetargetEnv
    from motrix_envs.locomotion.wbt.g1.omniretarget_fastsac import OmniRetargetFastSacEnv

    cfg = registry.make_env_config("g1-29dof-wbt-largebox-omniretarget-fastsac-a")
    cfg.commands.motion.start_at_timestep_zero_prob = 1.0
    env = OmniRetargetFastSacEnv(cfg, num_envs=1, seed=31)
    try:
        env.init_state()
        assert env._started_from_zero[0]
        motion = env.command_terms["motion"]
        assert motion.clip.joint_pos.shape[0] == 745

        # Place the physical box at the target final pose.
        writes = env.sim.compile_writes(
            {"p": BodyPositionWrite(("baseLink",)), "q": BodyRotationWrite(("baseLink",))}
        )
        writes.buffer("p")[0, 0] = motion.clip.extensions["object_pos_w"].data[-1]
        writes.buffer("q")[0, 0] = motion.clip.extensions["object_quat_w"].data[-1]
        writes.execute(np.array([0], dtype=np.int64))
        env._refresh_sim_reads()
        motion.clip_ended[0, 0] = True
        env.state.terminated[:] = False

        # Exercise the host-side bonus without conflating this focused check
        # with body-tracking termination when the robot remains at start pose.
        with patch.object(OmniRetargetEnv, "compute_transition", lambda self, state: state):
            state = env.compute_transition(env.state)
            # Being at the final box pose alone is not a successful carry.
            assert state.reward_terms["box_success_bonus"][0] == pytest.approx(0.0)
            assert not state.metrics["box_success"][0]
            assert not env._box_lift_verified[0]
            assert state.terminated[0]

            env._box_success_counted[:] = False
            state.terminated[:] = False
            motion.clip_ended[0, 0] = False
            reference = motion.clip.extensions["object_pos_w"].data
            stopped = np.zeros((1, 3), dtype=np.float32)

            # Less than 0.5 seconds in the elevated, reference-matched phase
            # must not count as a lift.
            for step in range(310, 334):
                motion.steps[0, 0] = step
                env._track_box_lift_and_setdown(motion, reference[step : step + 1], stopped)
            assert not env._box_lift_verified[0]
            motion.steps[0, 0] = 334
            env._track_box_lift_and_setdown(motion, reference[334:335], stopped)
            assert env._box_lift_verified[0]
            assert not env._box_setdown_verified[0]

            # Sustained lift without a later stable setdown still fails.
            motion.steps[0, 0] = 744
            motion.clip_ended[0, 0] = True
            state = env.compute_transition(state)
            assert state.reward_terms["box_success_bonus"][0] == pytest.approx(0.0)
            assert not state.metrics["box_success"][0]

            env._box_success_counted[:] = False
            state.terminated[:] = False
            motion.clip_ended[0, 0] = False
            for step in range(645, 660):
                motion.steps[0, 0] = step
                env._track_box_lift_and_setdown(motion, reference[step : step + 1], stopped)
            assert env._box_setdown_verified[0]

            motion.steps[0, 0] = 744
            motion.clip_ended[0, 0] = True
            state = env.compute_transition(state)
            assert state.reward_terms["box_success_bonus"][0] == pytest.approx(10.0)
            assert state.reward[0] == pytest.approx(10.0)
            assert state.metrics["box_success"][0]
            assert state.terminated[0]

            # Reaching the end again cannot double-count the terminal bonus.
            state.terminated[:] = False
            state = env.compute_transition(state)
            assert state.reward_terms["box_success_bonus"][0] == pytest.approx(0.0)

            # A random-start (non-full) episode cannot get the full completion bonus.
            env._box_success_counted[:] = False
            env._started_from_zero[:] = False
            state.terminated[:] = False
            state = env.compute_transition(state)
            assert state.reward_terms["box_success_bonus"][0] == pytest.approx(0.0)
    finally:
        del env
