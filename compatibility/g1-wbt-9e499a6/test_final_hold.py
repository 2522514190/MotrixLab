# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Regression tests executed against the isolated 9e499a6 source tree."""
import numpy as np
import pytest
from test_g1_wbt_task import _instant, _start_skill, _state, _task

from motrix_deploy.errors import ControlFailure
from motrix_deploy.runtime.context import ControlContext


def task_and_state(hold_s=.06):
    task = _task(controller=True, preparation=_instant, settings={
        'preparation': {'default_transition_duration_s': .02, 'kp': 10., 'kd': 1.},
        'final_frame_hold_s': hold_s,
    })
    state = _state()
    _start_skill(task, state)
    task.request_policy_start()
    task.request_motion_start()
    return task, state


def step(task, state, n):
    return task.step(state, ControlContext(n, n*.02, None, .02))


def test_final_reference_inference_budget_and_reset():
    task, state = task_and_state()
    observations = []
    original = task.policy_controller.policy.infer
    def infer(obs):
        observations.append(obs.copy())
        return original(obs)
    task.policy_controller.policy.infer = infer
    for n in range(6):
        assert step(task, state, n).command is not None
    assert task.phase == 'final_hold'
    assert task.playback_steps == 2
    assert task.policy_controller.completed_steps == 6  # 1 initial + 2 playback + 3 final.
    for obs in observations[3:]:
        np.testing.assert_allclose(obs[:29], task.motion.joint_pos[-1])
        np.testing.assert_allclose(obs[29:58], task.motion.joint_vel[-1])
    assert [x.name for x in task.transitions] == ['prepare', 'policy_hold', 'playback', 'final_hold']
    assert np.all(step(task, state, 6).command.kp == 0)
    assert task.phase == 'damping'
    task.reset(state)
    assert task.phase == 'prepare' and task.playback_steps == 0


def test_operator_stop_during_final_hold():
    task, state = task_and_state(600.)
    for n in range(4):
        step(task, state, n)
    assert task.phase == 'final_hold'
    task.request_stop()
    assert np.all(step(task, state, 4).command.kp == 0)
    assert task.phase == 'damping'


def test_final_hold_keeps_measured_termination_checks():
    task, state = task_and_state()
    for n in range(4):
        step(task, state, n)
    state.joint_velocity[0] = task.spec.termination_joint_velocity_threshold + 1
    with pytest.raises(ControlFailure, match='bad_dof_vel'):
        step(task, state, 4)


@pytest.mark.parametrize('value', [-1, True, float('nan'), float('inf'), '600'])
def test_invalid_hold_duration_is_rejected(value):
    with pytest.raises(ValueError, match='final_frame_hold_s'):
        task_and_state(value)


def test_six_hundred_second_budget_without_wall_clock_wait():
    task, state = task_and_state(599.96)  # Two clip frames = .04 s.
    for n in range(30001):
        result = step(task, state, n)
        assert result.command is not None and not result.complete
    assert task.phase == 'final_hold'
    assert task.policy_controller.completed_steps == 30001  # Plus mandatory initial hold.
    assert task.playback_steps == 2
    assert np.all(step(task, state, 30001).command.kp == 0)
    assert task.phase == 'damping'
