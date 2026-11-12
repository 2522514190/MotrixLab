"""Independent physical perturbation switches for robot-only WBT."""

from types import SimpleNamespace

import numpy as np

from motrix_env_core.manager import ManagerEnv
from motrix_envs.locomotion.wbt.g1.largebox import RobotOnlyCarryEnv


def test_disabling_push_preserves_other_dr_and_advances_physics(monkeypatch):
    env = object.__new__(RobotOnlyCarryEnv)
    env._cfg = SimpleNamespace(randomization=SimpleNamespace(enabled=True, push_enabled=False))
    env._push_remaining = np.asarray([-1.0, 0.0], dtype=np.float32)
    original = env._push_remaining.copy()
    calls = []
    monkeypatch.setattr(ManagerEnv, "physics_step", lambda self: calls.append(self))

    env.physics_step()

    assert calls == [env]
    assert env.cfg.randomization.enabled
    np.testing.assert_array_equal(env._push_remaining, original)
