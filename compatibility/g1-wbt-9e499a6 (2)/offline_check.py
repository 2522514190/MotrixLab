# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Offline artifact acceptance; no runtime, robot, DDS or LowCmd construction."""
import argparse
import importlib.abc
import json
from pathlib import Path
import socket
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--artifact', type=Path, required=True)
args = parser.parse_args()
root = args.source_root.resolve()
sys.path[:0] = [str(p) for p in sorted(root.glob('*/src'))]

class NoHardware(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('motrix_deploy_unitree', 'unitree_sdk2py')):
            raise RuntimeError(f'Hardware import forbidden: {fullname}')

sys.meta_path.insert(0, NoHardware())
class NoSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        raise RuntimeError('Network socket creation forbidden in offline check')
socket.socket = NoSocket

import numpy as np
from motrix_deploy.artifact.io import read_artifact
from motrix_deploy.contracts import RobotState
from motrix_deploy.policy import OnnxPolicy
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.task import create_task
from motrix_env_core.input import GamePadDevice

class Pad(GamePadDevice):
    def __init__(self):
        self.buttons = set()
        self.previous = set()
        self.down = set()
    def poll(self):
        self.down = self.buttons - self.previous
        self.previous = self.buttons.copy()
    def is_button_down(self, name):
        return name in self.down
    def is_button_pressing(self, name):
        return name in self.buttons
    def is_button_up(self, name):
        return False
    def axis_value(self, name):
        return 0.0

artifact = read_artifact(args.artifact.resolve())
m = artifact.manifest
policy = OnnxPolicy(artifact.policy_path, m.policy.input, m.policy.output)
settings = {'preparation': {'default_transition_duration_s': 0.04,
                           'wait_for_remote_buttons': True, 'start_button': 'start', 'enable_button': 'A'}}
task = create_task(m.task, m.robot, policy, artifact=artifact, settings=settings)
# Verify that source selection really uses the old runtime plus the backport.
import motrix_deploy.task as factory
import motrix_deploy_tasks.tasks.g1_wbt as wbt
assert Path(factory.__file__).resolve().is_relative_to(root)
assert Path(wbt.__file__).resolve().is_relative_to(root)
pad = Pad()
task.prepare(simulation=False, device=pad)
state = RobotState(0, 0, m.robot.default_joint_position.copy(), np.zeros(29, np.float32),
                   np.array([0, 0, 0, 1], np.float32), np.zeros(3, np.float32), np.zeros(3, np.float32))
task.reset(state)
def tick(n):
    return task.step(state, ControlContext(n, n * .02, None, .02))
for n in range(3):
    command = tick(n).command
    assert not np.any(command.kp) and not np.any(command.kd)
assert task.policy_controller.completed_steps == 0
pad.buttons = {'start', 'A'}
tick(3); tick(4)
assert task._startup.phase == 'wait_policy'
tick(5)
assert task._startup.phase == 'wait_policy'  # Early held A cannot confirm.
pad.buttons = set(); tick(6)
pad.buttons = {'A'}; tick(7)
assert task._startup.phase == 'complete'
assert task.phase == 'prepare' and not task.preparation_ready
assert task.policy_controller.completed_steps == 0
obs = task.policy_io.build_observation(state, ControlContext(0, 0, None, .02))
action = policy.infer(obs)
assert obs.shape == (154,) and action.shape == (29,) and np.isfinite(action).all()
for raw in (np.full(29, 1e6, np.float32), np.full(29, -1e6, np.float32), action):
    cmd = task.policy_io.process_action(raw)
    mid = (m.robot.position_lower + m.robot.position_upper) * .5
    half = (m.robot.position_upper - m.robot.position_lower) * (.5 * m.task.action_safety_clip_fraction)
    assert np.all(cmd.joint_position >= mid - half - 1e-6)
    assert np.all(cmd.joint_position <= mid + half + 1e-6)
    cmd.validate_for(m.robot)
# Unsupported settings must fail, rather than be silently ignored.
try:
    create_task(m.task, m.robot, policy, artifact=artifact, settings={'unknown': True})
except ValueError:
    pass
else:
    raise AssertionError('Unsupported settings were ignored')
print(json.dumps({'valid': True, 'motion_frames': len(task.motion), 'observation': list(obs.shape),
                  'action': list(action.shape), 'clip_fraction': m.task.action_safety_clip_fraction,
                  'startup': 'Start/ramp/fresh A verified', 'hardware_imports': False, 'network': False,
                  'factory_source': factory.__file__, 'task_source': wbt.__file__}, indent=2))
