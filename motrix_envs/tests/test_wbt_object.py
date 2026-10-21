# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0

"""Object tracking contracts: a shared clock, physical state, and privileged observations."""

from types import SimpleNamespace

import numpy as np
import pytest

from motrix_env_core import registry
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.sim import BodyPositionWrite
from motrix_envs.locomotion.wbt.mdp.object import (
    object_linear_velocity_obs,
    object_orientation_reward,
    object_position_obs,
    object_position_reward,
    object_tracking_failed,
)


def _context():
    identity = np.array([0, 0, 0, 1], dtype=np.float32)
    motion = SimpleNamespace(
        reference_index=0,
        steps=np.array([1]),
        clip=SimpleNamespace(
            extensions={
                "object_pos_w": SimpleNamespace(data=np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32)),
                "object_quat_w": SimpleNamespace(data=np.stack([identity, identity])),
            }
        ),
    )
    return SimpleNamespace(
        commands={"motion": motion},
        sim={
            "object_pos": np.array([1, 2, 3], dtype=np.float32),
            "object_quat": identity.copy(),
            "object_lin_vel": np.array([1, 0, 0], dtype=np.float32),
            "tracked_body_pos": np.array([[5, 6, 7]], dtype=np.float32),
            "tracked_body_quat": identity[None].copy(),
        },
        metrics={"object_position_error": np.zeros(1), "object_orientation_error": np.zeros(1)},
    )


def test_object_rewards_and_failure_use_physical_state_and_shared_frame():
    ctx = _context()
    assert object_position_reward(ctx, np.float32(1)) == pytest.approx(1)
    assert object_orientation_reward(ctx, np.float32(1)) == pytest.approx(1)
    assert not object_tracking_failed(ctx, np.float32(0.5), np.float32(0.5))
    ctx.sim["object_pos"][0] += 1
    assert object_position_reward(ctx, np.float32(1)) < 1
    assert object_tracking_failed(ctx, np.float32(0.5), np.float32(0.5))
    ctx.sim["object_pos"][:] = [1, 2, 3]
    ctx.sim["object_quat"][:] = [0, 0, 1, 0]
    assert object_orientation_reward(ctx, np.float32(1)) < 1
    assert object_tracking_failed(ctx, np.float32(0.5), np.float32(0.5))
    # q and -q describe the same rotation.
    ctx.sim["object_quat"][:] = [0, 0, 0, -1]
    assert object_orientation_reward(ctx, np.float32(1)) == pytest.approx(1)
    ctx.commands["motion"].steps[0] = 0
    assert object_position_reward(ctx, np.float32(1)) < 1


def test_object_observations_transform_points_and_vectors_differently():
    ctx = _context()
    # Reference is rotated +90 degrees around z.
    ctx.sim["tracked_body_quat"][0] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    out = np.empty(3, dtype=np.float32)
    object_position_obs(ctx, out)
    np.testing.assert_allclose(out, [-4, 4, -4], atol=1e-6)
    object_linear_velocity_obs(ctx, out)
    np.testing.assert_allclose(out, [0, -1, 0], atol=1e-6)
    ctx.sim["tracked_body_pos"][:] += 10
    object_linear_velocity_obs(ctx, out)
    np.testing.assert_allclose(out, [0, -1, 0], atol=1e-6)


@pytest.mark.numba
def test_compiled_object_reset_is_synchronized_and_lane_scoped():
    cfg = registry.make_env_config("g1-29dof-wbt-largebox-object", mode="play")
    env = ManagerEnv(cfg, num_envs=2)
    state = env.init_state()
    motion = env.command_terms["motion"]
    motion.steps[:, 0] = [motion.clip.joint_pos.shape[0] // 2, motion.clip.joint_pos.shape[0] - 1]
    env.reset(np.array([0, 1], dtype=np.int64))
    env._refresh_sim_reads()
    for key, channel in (("object_pos", "object_pos_w"), ("object_lin_vel", "object_lin_vel_w")):
        np.testing.assert_allclose(
            env.sim_data[key], motion.clip.extensions[channel].data[motion.steps[:, 0]], atol=1e-6
        )
    target_quat = motion.clip.extensions["object_quat_w"].data[motion.steps[:, 0]]
    np.testing.assert_allclose(np.abs(np.sum(env.sim_data["object_quat"] * target_quat, axis=-1)), 1, atol=1e-6)
    np.testing.assert_allclose(env.sim_data["robot_dof_pos"], motion.clip.joint_pos[motion.steps[:, 0]], atol=1e-6)
    untouched = env.sim_data["object_pos"][1].copy()
    motion.steps[0, 0] = 0
    env.reset(np.array([0], dtype=np.int64))
    env._refresh_sim_reads()
    np.testing.assert_allclose(env.sim_data["object_pos"][1], untouched)
    np.testing.assert_allclose(env.sim_data["object_pos"][0], motion.clip.extensions["object_pos_w"].data[0], atol=1e-6)
    # Only the critic receives the object's measured state.
    value_names = {term.name for term in env.observation_groups["value"].terms}
    policy_names = {term.name for term in env.observation_groups["policy"].terms}
    assert {"obj_pos_b", "obj_ori_b", "obj_lin_vel_b"} <= value_names
    assert {"obj_pos_b", "obj_ori_b", "obj_lin_vel_b"}.isdisjoint(policy_names)
    # Passing the clip end rematerializes both robot and object at the head.
    motion.steps[:, 0] = motion.clip.joint_pos.shape[0] - 1
    env.compute_transition(state)
    env._reset_done_envs()
    env._refresh_sim_reads()
    np.testing.assert_array_equal(motion.steps[:, 0], 0)
    np.testing.assert_allclose(
        env.sim_data["object_pos"], np.broadcast_to(motion.clip.extensions["object_pos_w"].data[0], (2, 3)), atol=1e-6
    )
    state = env.step(np.zeros((2, *env.action_space.shape), dtype=np.float32))
    assert np.all(np.isfinite(state.reward))
    assert np.all(np.isfinite(state.obs.value))
    # Raw diagnostics reach the same metrics view consumed by the training logger.
    from motrix_envs.locomotion.wbt.mdp.tracking_metrics import LargeBoxTrackingMetricsCfg

    for name in LargeBoxTrackingMetricsCfg()(None).metric_names:
        assert name in state.metrics
        assert np.all(np.isfinite(state.metrics[name]))
        assert np.all(state.metrics[name] >= 0)
    assert not np.any(state.metrics["tracking_metrics"])

    # Move the real box independently of the reference. Evaluating tracking
    # must report the deviation without overwriting its physical pose.
    motion.steps[:, 0] = 0
    env.reset(np.array([0, 1], dtype=np.int64))
    env._refresh_sim_reads()
    displacement = cfg.terminations.object_tracking.position_threshold * 2
    writes = env.sim.compile_writes({"position": BodyPositionWrite((cfg.sim_reset.object_state.body,))}, reset=False)
    moved = env.sim_data["object_pos"][0].copy()
    moved[0] += displacement
    writes.buffer("position")[0, 0] = moved
    writes.execute(np.array([0], dtype=np.int64))
    env._refresh_sim_reads()
    env._execute_task_kernel(env._kernel_inputs)
    assert env.metrics["object_position_error"][0, 0] > cfg.terminations.object_tracking.position_threshold
    np.testing.assert_allclose(env.sim_data["object_pos"][0], moved, atol=1e-6)


@pytest.mark.numba
def test_object_startup_mass_mixture_scales_inertia_and_survives_resets():
    cfg = registry.make_env_config("g1-29dof-wbt-largebox-object")
    # Exercise the contract with test-specific ranges and inertia ratios.
    randomization = cfg.sim_reset.object_mass
    randomization.light_probability = 0.75
    randomization.light_range = (0.4, 0.8)
    randomization.heavy_range = (1.2, 1.8)
    randomization.inertia_per_mass = (0.03, 0.04, 0.05, 0.0, 0.0, 0.0)
    env = ManagerEnv(cfg, num_envs=128, seed=31)
    env.init_state()
    link = env.sim._model.get_link(randomization.body)
    data = env.sim._data

    def read():
        return np.array(link.get_mass_override(data)).reshape(-1), np.array(link.get_inertia_override(data))

    mass, inertia = read()
    light = (mass >= 0.4) & (mass <= 0.8)
    heavy = (mass >= 1.2) & (mass <= 1.8)
    assert np.all(light | heavy)
    assert 0.55 < light.mean() < 0.95
    np.testing.assert_allclose(inertia, mass[:, None] * randomization.inertia_per_mass, rtol=1e-6, atol=1e-8)
    selected = np.array([1, 4, 11], dtype=np.int64)
    untouched = np.setdiff1d(np.arange(env.num_envs), selected)
    env.reset(selected)
    after_mass, after_inertia = read()
    np.testing.assert_array_equal(after_mass[untouched], mass[untouched])
    np.testing.assert_array_equal(after_inertia[untouched], inertia[untouched])
    np.testing.assert_array_equal(after_mass[selected], mass[selected])
    np.testing.assert_array_equal(after_inertia[selected], inertia[selected])
    np.testing.assert_allclose(
        after_inertia, after_mass[:, None] * randomization.inertia_per_mass, rtol=1e-6, atol=1e-8
    )
    env.physics_step()
    physical_mass, physical_inertia = read()
    np.testing.assert_array_equal(physical_mass, after_mass)
    np.testing.assert_array_equal(physical_inertia, after_inertia)


@pytest.mark.numba
def test_object_play_mass_is_fixed_across_resets():
    cfg = registry.make_env_config("g1-29dof-wbt-largebox-object", mode="play")
    # A fixed evaluation target must not depend on the mixture branch.
    cfg.sim_reset.object_mass.light_range = (1.3, 1.3)
    cfg.sim_reset.object_mass.heavy_range = (1.3, 1.3)
    env = ManagerEnv(cfg, num_envs=3, seed=9)
    env.init_state()
    link = env.sim._model.get_link(cfg.sim_reset.object_mass.body)
    for _ in range(2):
        env.reset(np.arange(3, dtype=np.int64))
        np.testing.assert_allclose(link.get_mass_override(env.sim._data), 1.3)
        np.testing.assert_allclose(
            np.array(link.get_inertia_override(env.sim._data)),
            np.tile(np.array(cfg.sim_reset.object_mass.inertia_per_mass) * 1.3, (3, 1)),
            rtol=1e-6,
        )
