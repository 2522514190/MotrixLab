# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Coupled mass/inertia draws and staged dynamics perturbations."""

from types import SimpleNamespace

import numpy as np

from motrix_envs.locomotion.wbt.g1.largebox import RobotOnlyCarryEnv, RobotOnlyRandomizationCfg


class _Program:
    def __init__(self, writes, num_envs):
        self.buffers = {}
        for key, write in writes.items():
            if key in ("kp", "kd"):
                shape = (num_envs, len(write.actuators))
            else:
                shape = (num_envs, len(write.links))
                if key in ("inertia", "com"):
                    shape += (6 if key == "inertia" else 3,)
            self.buffers[key] = np.empty(shape, dtype=np.float32)

    def buffer(self, key):
        return self.buffers[key]


def _environment(dr):
    env = object.__new__(RobotOnlyCarryEnv)
    env._cfg = SimpleNamespace(randomization=dr)
    env._num_envs = 16
    env.model = SimpleNamespace(
        bodies={"robot": SimpleNamespace(link_names=("base", "tip", "massless"))},
        actuators=(SimpleNamespace(name="a"), SimpleNamespace(name="b")),
        others={
            "robot_link_masses": np.array([2.0, 0.5, 0.0]),
            "robot_link_inertias": np.array([[2, 3, 4, 0.1, 0, 0], [1, 1, 1, 0, 0, 0], [0] * 6]),
            "torso_com": np.array([0.1, 0.2, 0.3]),
            "actuator_kp": np.array([10.0, 20.0]),
            "actuator_kd": np.array([1.0, 2.0]),
        },
    )
    env.sim = SimpleNamespace(compile_writes=lambda writes, **kwargs: _Program(writes, env.num_envs))
    return env


def test_mass_and_full_inertia_share_per_link_factors_and_staged_draws_are_stable():
    mass_env = _environment(RobotOnlyRandomizationCfg(link_mass_scale=(0.7, 1.3)))
    mass = mass_env._make_domain_writes(17)
    factors = mass.buffer("mass") / mass_env.model.others["robot_link_masses"][:2]
    assert np.all((factors >= 0.7) & (factors <= 1.3))
    assert not np.allclose(factors[:, 0], factors[:, 1])
    np.testing.assert_allclose(
        mass.buffer("inertia"), factors[..., None] * mass_env.model.others["robot_link_inertias"][:2], atol=1e-6
    )
    dr = RobotOnlyRandomizationCfg(
        link_mass_scale=(0.7, 1.3), torso_com_offset=(0.01, 0.03, 0.02), kp_scale=(0.8, 1.2), kd_scale=(0.8, 1.2)
    )
    full_env = _environment(dr)
    full = full_env._make_domain_writes(17)
    np.testing.assert_array_equal(full.buffer("mass"), mass.buffer("mass"))
    offset = full.buffer("com")[:, 0] - full_env.model.others["torso_com"]
    assert np.all(np.abs(offset) <= np.array(dr.torso_com_offset) + 1e-7)
    kp = full.buffer("kp") / full_env.model.others["actuator_kp"]
    kd = full.buffer("kd") / full_env.model.others["actuator_kd"]
    assert np.all((kp >= 0.8) & (kp <= 1.2))
    assert np.all((kd >= 0.8) & (kd <= 1.2))
    assert not np.allclose(kp, kd)


def test_disabled_dynamics_randomization_keeps_nominal_properties():
    env = _environment(RobotOnlyRandomizationCfg(enabled=False, link_mass_scale=(0.7, 1.3)))
    assert env._make_domain_writes(17) is None
    env = _environment(RobotOnlyRandomizationCfg())
    assert env._make_domain_writes(17) is None
