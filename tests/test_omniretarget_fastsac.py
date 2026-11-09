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
        assert env.cfg.object_curriculum.reference_frames == 745
        assert env.command_terms["motion"].clip.joint_pos.shape[0] == 745
        assert env.cfg.commands.motion.motion_files[0].endswith("sub3_largebox_003_unilab_grounded_smooth_745.npz")
        assert not env._gate.enabled[0]
        indices = env._box_weight_indices
        np.testing.assert_allclose(env._task_program.reward_weights[list(indices)], 0.5)
        assert "box_success_bonus" in env.state.reward_terms
        assert "box_success" in env.state.metrics

        # Curriculum is performance-triggered and independent of the error gate.
        env._gate.enabled[0] = False
        env._elapsed_since_reset[:] = 100
        env._on_full_start_episodes_finished(np.array([0, 1], dtype=np.int64))
        assert env._box_reward_weight == pytest.approx(0.5)
        for _ in range(256):
            env._elapsed_since_reset[:] = env.cfg.object_curriculum.reference_frames
            env._on_full_start_episodes_finished(np.array([0, 1], dtype=np.int64))
        assert env._box_curriculum_enabled
        assert env._box_reward_weight < 0.5
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
