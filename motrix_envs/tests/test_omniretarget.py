# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0

"""Behavior contracts for calibrated control and staged tracking."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from motrix_env_core import registry
from motrix_env_core.sim.write import BodyPositionWrite
from motrix_envs.locomotion.wbt.g1.omniretarget import OmniRetargetEnv
from motrix_envs.locomotion.wbt.mdp.omniretarget import (
    CalibratedPositionParams,
    ObjectTerminationGate,
    calibrated_joint_positions,
    relative_rot6d_columns,
    staged_object_failure,
)


def test_rot6d_uses_first_two_columns_for_nontrivial_rotation():
    q = np.array([0.2, -0.4, 0.1, 0.8], dtype=np.float32)
    q /= np.linalg.norm(q)
    out = np.empty(6, dtype=np.float32)
    relative_rot6d_columns(np.array([0, 0, 0, 1], dtype=np.float32), q, out)
    from scipy.spatial.transform import Rotation

    expected = Rotation.from_quat(q).as_matrix()[:, :2].reshape(-1)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_calibration_observation_uses_same_bias_as_action_in_joint_order():
    # Arbitrary test-specific offsets and a non-identity actuator permutation.
    action = SimpleNamespace(bias=np.array([0.12, -0.07], dtype=np.float32))
    ctx = SimpleNamespace(
        actions={"joint_position": action},
        sim={"robot_dof_pos": np.array([0.4, 0.8], dtype=np.float32)},
        rand=SimpleNamespace(state=np.array([42], dtype=np.uint64)),
    )
    params = CalibratedPositionParams(np.array([0.1, 0.2], dtype=np.float32), np.array([1, 0]), np.float32(0))
    out = np.empty(2, dtype=np.float32)
    calibrated_joint_positions(ctx, out, params)
    np.testing.assert_allclose(out, [0.37, 0.48], atol=1e-6)


def test_omniretarget_uses_shared_undesired_contact_query():
    cfg = registry.make_env_config("g1-29dof-wbt-largebox-omniretarget")
    assert not tuple(cfg.scene.sensors.items())
    assert "undesired_contact_forces" in cfg.queries.data
    assert type(cfg.rewards.undesired_contacts).__name__ == "UndesiredContactsRewardCfg"


def test_object_gate_suppresses_failure_but_keeps_diagnostics_and_uses_or():
    q = np.array([0, 0, 0, 1], dtype=np.float32)
    ctx = SimpleNamespace(
        commands={
            "motion": SimpleNamespace(
                steps=np.array([0]),
                clip=SimpleNamespace(
                    extensions={
                        "object_pos_w": SimpleNamespace(data=np.zeros((1, 3), dtype=np.float32)),
                        "object_quat_w": SimpleNamespace(data=q[None]),
                    }
                ),
            )
        },
        sim={"object_pos": np.array([2, 0, 0], dtype=np.float32), "object_quat": q.copy()},
        metrics={
            name: np.zeros(1, dtype=np.float32)
            for name in ("object_position_error", "object_orientation_error", "object_termination_enabled")
        },
    )
    gate = ObjectTerminationGate(np.array([False]))
    assert not staged_object_failure(ctx, gate, np.float32(0.6), np.float32(0.7))
    assert ctx.metrics["object_position_error"][0] == 2
    gate.enabled[0] = True
    assert staged_object_failure(ctx, gate, np.float32(0.6), np.float32(0.7))
    ctx.sim["object_pos"][:] = 0
    ctx.sim["object_quat"][:] = [0, 0, math.sin(0.5), math.cos(0.5)]
    assert staged_object_failure(ctx, gate, np.float32(0.6), np.float32(0.7))


@pytest.fixture(scope="module")
def paper_env():
    cfg = registry.make_env_config("g1-29dof-wbt-largebox-omniretarget")
    # Test ranges exercise the randomization contract without pinning task tuning.
    cfg.randomization.object_mass = (0.4, 0.8)
    cfg.randomization.object_com = (0.01, 0.02, 0.03)
    cfg.randomization.torso_com = (0.02, 0.03, 0.01)
    cfg.randomization.object_inertia_scale = (0.7, 1.3)
    env = OmniRetargetEnv(cfg, num_envs=8, seed=97)
    env.init_state()
    return env


@pytest.mark.numba
def test_physical_randomization_survives_partial_and_clip_resets(paper_env):
    env = paper_env
    data = env.sim._data
    model = env.sim._model
    link = model.get_link("baseLink")
    mass = np.array(link.get_mass_override(data)).reshape(-1)
    inertia = np.array(link.get_inertia_override(data))
    np.testing.assert_array_less(0.3999, mass)
    np.testing.assert_array_less(mass, 0.8001)
    factor = inertia[:, 0] / (mass * 0.02)
    assert np.all((factor >= 0.7) & (factor <= 1.3))
    com = np.array(link.get_center_of_mass_override(data))
    assert np.all(np.abs(com) <= env.cfg.randomization.object_com)
    torso = np.array(model.get_link("torso_link").get_center_of_mass_override(data))
    # A partial episode reset plus an independent clip rematerialization.
    env.reset(np.array([1, 4], dtype=np.int64), np.array([3], dtype=np.int64))
    np.testing.assert_array_equal(np.array(link.get_mass_override(data)).reshape(-1), mass)
    np.testing.assert_array_equal(np.array(link.get_inertia_override(data)), inertia)
    np.testing.assert_array_equal(np.array(link.get_center_of_mass_override(data)), com)
    retained = [0, 2, 3, 5, 6, 7]
    np.testing.assert_array_equal(env.action_terms["joint_position"].state.bias, 0.0)
    updated_torso = np.array(model.get_link("torso_link").get_center_of_mass_override(data))
    np.testing.assert_array_equal(updated_torso[retained], torso[retained])
    assert np.all(np.any(updated_torso[[1, 4]] != torso[[1, 4]], axis=1))
    # FastSAC uses the resolved per-joint finite policy bounds.
    assert np.all(np.isfinite(env.action_space.low))
    assert np.all(np.isfinite(env.action_space.high))
    assert np.all(env.action_space.low < 0)
    assert np.all(env.action_space.high > 0)


@pytest.mark.numba
@pytest.mark.parametrize("curriculum_enabled,evaluation", [(False, False), (True, False), (True, True)])
def test_object_termination_activation_and_optional_curriculum_latch(curriculum_enabled, evaluation):
    cfg = registry.make_env_config("g1-29dof-wbt-largebox-omniretarget")
    cfg.object_curriculum.enabled = curriculum_enabled
    cfg.evaluation = evaluation
    env = OmniRetargetEnv(cfg, num_envs=2, seed=97)
    env.init_state()
    policy = {term.name for term in env.observation_groups["policy"].terms}
    value = {term.name for term in env.observation_groups["value"].terms}
    assert {"motion_ref_pos_b", "base_lin_vel"}.isdisjoint(policy)
    assert {"motion_ref_pos_b", "base_lin_vel"} <= value
    assert {"obj_pos_b", "obj_ori_b", "obj_lin_vel_b"} <= value
    assert {"obj_pos_b", "obj_ori_b", "obj_lin_vel_b"}.isdisjoint(policy)
    gate = env._gate
    initially_active = evaluation or not curriculum_enabled
    assert bool(gate.enabled[0]) == initially_active
    # A failed object must terminate immediately unless training explicitly
    # requests the delayed curriculum; evaluation always checks object failure.
    env._refresh_sim_reads()
    writer = env.sim.compile_writes({"p": BodyPositionWrite(("baseLink",))})
    writer.buffer("p")[0, 0] = env.sim_data["object_pos"][0] + [4, 0, 0]
    writer.execute(np.array([0], dtype=np.int64))
    env._refresh_sim_reads()
    env._execute_evaluate_kernel(env._kernel_inputs)
    assert bool(env.state.metrics["object_tracking"][0]) == initially_active
    env.reset(np.arange(env.num_envs, dtype=np.int64))
    if initially_active:
        assert gate.enabled[0]
        assert env._average_episode_steps == 0
        return
    # Use a short test horizon gate; only real completed episodes count.
    env.cfg.object_curriculum.episode_window = 2
    env.cfg.object_curriculum.enable_at_horizon_fraction = 0.5
    env._average_episode_steps = 0
    env._elapsed_since_reset[:] = env.cfg.max_episode_steps
    env.state.terminated[:] = False
    env.state.truncated[:] = False
    env.reset(np.array([0, 1], dtype=np.int64))
    assert not gate.enabled[0]
    env._elapsed_since_reset[:] = env.cfg.max_episode_steps
    env.state.truncated[:2] = True
    env.reset(np.array([0, 1], dtype=np.int64))
    assert gate.enabled[0]
    env._elapsed_since_reset[:] = 1
    env.reset(np.array([0, 1], dtype=np.int64))
    assert gate.enabled[0]
    env.state.truncated[:] = False
    # Move the actual object without moving its reference; active gate sees it.
    env._refresh_sim_reads()
    writer.buffer("p")[0, 0] = env.sim_data["object_pos"][0] + [4, 0, 0]
    writer.execute(np.array([0], dtype=np.int64))
    env._refresh_sim_reads()
    env._execute_evaluate_kernel(env._kernel_inputs)
    assert env.state.metrics["object_tracking"][0]


@pytest.mark.numba
def test_push_schedule_is_lane_scoped_and_adds_velocity_without_teleport(paper_env):
    env = paper_env
    env.reset(np.arange(env.num_envs, dtype=np.int64))
    env._refresh_sim_reads()
    position = env.sim_data["object_pos"].copy()
    env._push_remaining[:] = 10
    env._push_remaining[2] = 0
    before = env.sim_data["root_lin_vel"].copy()
    env.physics_step()
    # The stored write is the instantaneous impulse before physics integration.
    written = env._push_writes.buffer("linear")[2, 0]
    assert np.any(written != before[2])
    assert np.all(np.abs(written - before[2]) <= env._push_linear + 1e-6)
    assert np.all(env._push_remaining[np.arange(env.num_envs) != 2] > 9)
    env._refresh_sim_reads()
    assert np.max(np.linalg.norm(env.sim_data["object_pos"] - position, axis=-1)) < 0.1
