# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the versioned 506-frame physical carry task."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from motrix_env_core import registry
from motrix_envs.locomotion.wbt.g1.opentrack506 import TASK, OpenTrackCarryEnv, self_contact_count
from motrix_envs.locomotion.wbt.mdp.observations import reference_body_orientation_obs


@pytest.mark.parametrize("rotations", [Rotation.identity(14), Rotation.random(14, random_state=42)])
def test_reference_body_observation(rotations):
    quats = rotations.as_quat().astype(np.float32)
    ctx = SimpleNamespace(sim={"tracked_body_quat": quats}, commands={"motion": SimpleNamespace(reference_index=7)})
    actual = np.empty(84, np.float32)
    reference_body_orientation_obs(ctx, actual)
    expected = rotations[7].as_matrix().T @ rotations.as_matrix()
    np.testing.assert_allclose(actual.reshape(14, 2, 3), expected[:, :2], atol=1e-6)


def test_robot_only_contact_pairs_count_each_eligible_body_once():
    cfg = registry.make_env_config(TASK)
    base = registry.make_env_config("g1-29dof-wbt-largebox-opentrack406-omni-reward-push")
    assert not base.observations.value.robot_body_ori_b.corrected_reference
    assert cfg.observations.value.robot_body_ori_b.corrected_reference
    assert cfg.observations.policy == base.observations.policy
    assert cfg.actions == base.actions
    assert cfg.commands.motion.kernel_size == base.commands.motion.kernel_size
    pairs = list(cfg.scene.sensors.pairs.values())
    assert all("floor" not in (p.geom1, p.geom2) and "largebox_collision" not in (p.geom1,p.geom2) for p in pairs)
    owners = np.array(cfg.rewards.undesired_contacts.owners)
    forces = np.zeros((len(pairs), 3), np.float32)
    ctx = SimpleNamespace(sim={"self_contact_forces": forces.ravel()})
    assert self_contact_count(ctx, owners, 1.) == 0
    forces[:2, 0] = 2
    assert self_contact_count(ctx, owners, 1.) == len(set(owners[:2].ravel()) - {-1})


@pytest.mark.numba
def test_carry_runtime_reset_and_termination():
    cfg = registry.make_env_config(TASK)
    env = OpenTrackCarryEnv(cfg, num_envs=4, seed=19)
    state = env.init_state()
    assert state.obs.policy.shape == (4, 154)
    assert state.obs.value.shape == (4, 298)
    motion = env.command_terms["motion"]
    assert len(motion.clip.joint_pos) == 506
    assert cfg.max_episode_steps == 550
    assert cfg.sim_reset.object_state.noise == 0
    mass = np.array(env.sim._model.get_link("baseLink").get_mass_override(env.sim._data))
    assert np.all((mass >= 0.1) & (mass <= 0.5))
    empty = np.empty(0, dtype=np.int64)
    env.reset(empty, np.array([2]))
    env._refresh_sim_reads()
    np.testing.assert_allclose(env.sim_data["object_pos"][2], motion.clip.extensions["object_pos_w"].data[motion.steps[2,0]], atol=1e-5)
    np.testing.assert_allclose(env.sim_data["object_ang_vel"][2], 0., atol=1e-5)
    np.testing.assert_array_equal(mass, env.sim._model.get_link("baseLink").get_mass_override(env.sim._data))
    env._average_episode_steps = 439.9
    env._elapsed_since_reset[:] = 550
    env.reset(empty, np.array([1]))
    assert not env._gate.enabled[0]
    state.terminated[:] = True
    env.reset(np.arange(4))
    assert env._gate.enabled[0]
