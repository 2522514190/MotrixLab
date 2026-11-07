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
    assert np.all(np.isneginf(term.action_space.low))
    assert np.all(np.isposinf(term.action_space.high))
    actions = np.broadcast_to(bounds * 0.2, (env.num_envs, bounds.size)).copy()
    targets = term.process(actions)
    np.testing.assert_allclose(targets, state.default_angles + state.bias + actions * state.action_scales, atol=1e-6)
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
    policy, critic, rewards, terminated, truncated = wrapped_env.step(actions)
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
    _, _, _, terminated, truncated = wrapped_env.step(torch.zeros((env.num_envs, env.action_space.shape[0])))
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
        assert not env._gate.enabled[0]
        indices = env._box_weight_indices
        np.testing.assert_allclose(env._task_program.reward_weights[list(indices)], 0.5)
        assert "box_success_bonus" in env.state.reward_terms
        assert "box_success" in env.state.metrics

        # Forcing the already-compiled gate demonstrates that the runtime
        # weights (not only cfg.rewards.*.weight) actually change.
        env._gate.enabled[0] = True
        env._box_decay_steps = 2
        env.step(np.zeros((env.num_envs, env.action_space.shape[0]), dtype=np.float32))
        np.testing.assert_allclose(env._task_program.reward_weights[list(indices)], 0.35, atol=1e-6)
        env.step(np.zeros((env.num_envs, env.action_space.shape[0]), dtype=np.float32))
        np.testing.assert_allclose(env._task_program.reward_weights[list(indices)], 0.2, atol=1e-6)
        assert env.state.metrics["box_reward_weight"] == pytest.approx(0.2)
    finally:
        env.close()


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
        env.close()
