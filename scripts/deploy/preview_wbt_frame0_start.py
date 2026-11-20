# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo-only preview: R holds frame 0; A starts policy playback; O stops."""
from __future__ import annotations

import argparse
import json

import numpy as np

from motrix_deploy.artifact.io import read_artifact
from motrix_deploy.config import SimRuntimeCfg
from motrix_deploy.contracts import JointServoCommand
from motrix_deploy.policy import OnnxPolicy
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.runtime.control import ControlSession
from motrix_deploy.runtime.factory import create_simulation_runtime
from motrix_deploy.runtime.lifecycle import ControllerStep
from motrix_deploy_tasks.tasks.g1_wbt import G1WbtDeployTask


class FrameZeroPreview(G1WbtDeployTask):
    """Manual simulation takeover without the automatic preparation tilt gate.

    Original policy termination checks remain active after A. This preview has
    no hardware entry point and does not modify the artifact or PC2 runtime.
    """

    def __init__(self, *args, auto=False, preview_s=None, **kwargs):
        super().__init__(*args, automatic_start=False, **kwargs)
        self.auto = auto
        self.preview_s = preview_s
        self.device = None

    def prepare(self, *, simulation, device=None, spawn_pose=None):
        if not simulation:
            raise ValueError('This manual takeover preview is only supported in simulation')
        self.device = device

    def reset(self, state):
        super().reset(state)
        self._enter_phase('wait_start')
        self._initial_hold = state.joint_position.copy()
        self._ramp_start = None
        self._policy_start_s = None
        self._last_report_s = -1.
        print('R = Start: move to frame 0 and hold; A = start policy; O = stop', flush=True)

    def step(self, state, context):
        start = enable = stop = False
        if self.device is not None:
            self.device.poll()
            start = self.device.is_key_down('r')
            enable = self.device.is_key_down('a')
            stop = self.device.is_key_down('o')
        if self.auto:
            start = context.elapsed_time_s >= .2
            enable = self._ramp_start is not None and context.elapsed_time_s >= self._ramp_start + 3.
        if self.preview_s is not None and self._policy_start_s is not None:
            stop |= context.elapsed_time_s >= self._policy_start_s + self.preview_s
        if stop:
            self.request_stop()
        if self._stop_requested:
            return super().step(state, context)
        if self._phase == 'wait_start' and start:
            self._initial_hold = state.joint_position.copy()
            self._ramp_start = context.elapsed_time_s
            self._enter_phase('frame0_hold')
            print('Start accepted: moving to frame 0 with artifact preparation gains', flush=True)
        if self._phase in {'wait_start', 'frame0_hold'}:
            fraction = 0.
            target = self._initial_hold
            if self._phase == 'frame0_hold':
                duration = self.spec.preparation_ramp_duration_s
                fraction = min((context.elapsed_time_s - self._ramp_start + context.dt_s) / duration, 1.) if duration else 1.
                target = self._initial_hold + fraction * (self.motion.joint_pos[0] - self._initial_hold)
            x, y, _, _ = state.base_orientation_xyzw
            tilt = float(np.arccos(np.clip(1 - 2 * (float(x)**2 + float(y)**2), -1, 1)))
            if context.elapsed_time_s - self._last_report_s >= .5:
                print(json.dumps(dict(phase=self._phase, tilt_deg=float(np.rad2deg(tilt)),
                    joint_error_rad=float(np.max(np.abs(state.joint_position - self.motion.joint_pos[0]))),
                    max_joint_velocity=float(np.max(np.abs(state.joint_velocity))))), flush=True)
                self._last_report_s = context.elapsed_time_s
            if self._phase == 'frame0_hold' and fraction >= 1. and enable:
                self.policy_io.align_reference_to(state)
                self.policy_controller.reset(state)
                self.policy_controller.steps = self._playback_budget
                self._policy_start_s = context.elapsed_time_s
                self._enter_phase('playback')
                print('A accepted: policy starts at frame 0; policy termination checks active', flush=True)
            else:
                zeros = np.zeros(29, np.float32)
                return ControllerStep(command=JointServoCommand(
                    np.clip(target, self.robot.position_lower, self.robot.position_upper).astype(np.float32),
                    zeros, zeros, self._preparation_kp, self._preparation_kd))
        return super().step(state, context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--auto', action='store_true', help='Automatically request R then A for numerical evaluation')
    parser.add_argument('--preview-s', type=float, help='Stop after this many seconds of policy control')
    parser.add_argument('--push', action='store_true', help='Enable the runtime velocity disturbances during policy control')
    parser.add_argument('--randomization', action='store_true', help='Enable initial-state randomization and observation noise')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    if args.headless and not args.auto:
        parser.error('--headless requires --auto')
    artifact = read_artifact(args.artifact)
    m = artifact.manifest
    policy = OnnxPolicy(artifact.policy_path, m.policy.input, m.policy.output)
    task = FrameZeroPreview(m.task, m.robot, policy, artifact=artifact, auto=args.auto, preview_s=args.preview_s)
    cfg = SimRuntimeCfg(backend='mujoco', deploy_env_id='flat', robot_id='g1-29dof',
                       viewer=not args.headless, floor_sliding_friction=1.4, box_enabled=False)
    cfg.randomization.enabled = args.randomization
    cfg.randomization.seed = args.seed
    cfg.push.enabled = args.push
    cfg.push.seed = args.seed
    print(json.dumps(dict(push=vars(cfg.push), randomization=vars(cfg.randomization))), flush=True)
    cfg.sensor_bindings.base_angular_velocity = 'base_gyro'
    cfg.sensor_bindings.base_linear_acceleration = 'base_accel'
    runtime = create_simulation_runtime('mujoco', task.configure_simulation(cfg.build_runtime_config()))
    task.prepare(simulation=True, device=runtime.keyboard)
    session = ControlSession(controller=task, robot=runtime.robot,
                             period_s=m.control.period_s, state_timeout_s=m.control.state_timeout_s)
    runtime.bind_session(session)
    with runtime:
        result = runtime.run()
        print(json.dumps(dict(result=result.to_dict(), playback_frames=task.playback_steps), indent=2), flush=True)


if __name__ == '__main__':
    main()
