# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/motion/evaluate_largebox.py"
SPEC = importlib.util.spec_from_file_location("evaluate_largebox", SCRIPT)
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


def test_initialization_resets_but_rollout_never_recovers(monkeypatch):
    calls = []
    monkeypatch.setattr(evaluate.ManagerEnv, "_reset_done_envs", lambda self: calls.append("reset"))

    def initialize(self):
        self._reset_done_envs()
        return "state"

    monkeypatch.setattr(evaluate.ManagerEnv, "init_state", initialize)
    env = object.__new__(evaluate.NoRecoveryEnv)
    assert env.init_state() == "state"
    env._reset_done_envs()
    assert calls == ["reset"]


@pytest.mark.parametrize("failure_step,success,steps", [(2, False, 2), (5, False, 5), (None, True, 5)])
@pytest.mark.parametrize("with_object", [False, True])
def test_first_failure_ends_trial_including_final_frame(failure_step, success, steps, with_object):
    class FakeEnv:
        def __init__(self):
            self.calls = 0
            self.motion = SimpleNamespace(steps=np.zeros((1, 1), dtype=np.int64))
            metrics = {
                "bad_body_z": np.ones(1),
            }
            if with_object:
                metrics.update(
                    object_position_error=np.array([0.1]),
                    object_orientation_error=np.array([0.2]),
                )
            self.env = SimpleNamespace(
                command_terms={"motion": self.motion},
                cfg=SimpleNamespace(terminations=SimpleNamespace(to_dict=lambda: {"bad_body_z": None})),
                _state=SimpleNamespace(metrics=metrics),
            )

        def reset(self):
            self.calls = 0
            self.motion.steps.fill(0)
            return torch.zeros(1), None

        def step(self, action):
            self.calls += 1
            self.motion.steps[0, 0] = self.calls
            return torch.zeros(1), None, None, torch.tensor([self.calls == failure_step]), torch.tensor([False])

    env = FakeEnv()
    agent = SimpleNamespace(act=lambda obs, deterministic: torch.zeros(1))
    result = evaluate.evaluate_episode(env, agent, num_frames=5)
    assert result["success"] is success
    assert result["steps"] == env.calls == steps
    assert result["reasons"] == ([] if success else ["bad_body_z"])
    assert ("object_position_error_m" in result) is (with_object and not success)
