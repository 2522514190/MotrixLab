# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0

"""Paper-specific observations, calibrated actions and staged object termination.

Sources: OmniRetarget IV; BeyondMimic S1; Holosoma WBT PPO where unspecified.
These terms do not inherit the tuned FastSAC task's settings.
"""

import math
from dataclasses import fields

import numpy as np
from numba import njit

from motrix_env_core.config import configclass
from motrix_env_core.manager import (
    ManagerContext,
    ObservationTermCfg,
    ObsTerm,
    TerminationTerm,
    TerminationTermCfg,
)
from motrix_env_core.manager.math.quaternion import inverse, mul, rotation_distance
from motrix_env_core.mdp.action import JointPositionActionCfg, JointPositionActionState, JointPositionActionTerm
from motrix_env_core.mdp.noise import add_uniform_noise
from motrix_env_core.numba.kernel_data import SharedArray, kernel_data
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_envs.locomotion.wbt.mdp.observations import motion_reference_position_obs


@kernel_data
class CalibratedActionState(JointPositionActionState):
    bias: np.ndarray


class CalibratedActionTerm(JointPositionActionTerm):
    def __init__(self, action_space, state, safety_clip_fraction: float, clip_target: bool = True):
        super().__init__(action_space, state)
        self._safety_clip_fraction = np.float32(safety_clip_fraction)
        self._clip_target = bool(clip_target)

    def _process(self, actions: np.ndarray) -> np.ndarray:
        state = self.state
        target = super()._process(actions) + state.bias
        if not self._clip_target:
            return target
        midpoint = (state.joint_lower + state.joint_upper) * 0.5
        half_range = (state.joint_upper - state.joint_lower) * (0.5 * self._safety_clip_fraction)
        return np.clip(target, midpoint - half_range, midpoint + half_range)


@configclass(kw_only=True)
class CalibratedActionCfg(JointPositionActionCfg):
    """Calibration bias affects both PD setpoints and joint observations (BM S1)."""

    safety_clip_fraction: float = 1.0
    # Opt out only for explicitly configured ablations; legacy recipes retain
    # the calibrated safety clipping on their PD position setpoints.
    clip_target: bool = True

    def __call__(self, env, actuators):
        if not 0.0 < self.safety_clip_fraction <= 1.0:
            raise ValueError("safety_clip_fraction must be in (0, 1].")
        base = super().__call__(env, actuators)
        state = CalibratedActionState(
            **{item.name: getattr(base.state, item.name) for item in fields(JointPositionActionState)},
            bias=np.zeros((env.num_envs, base.action_space.shape[0]), dtype=np.float32),
        )
        return CalibratedActionTerm(base.action_space, state, self.safety_clip_fraction, self.clip_target)


@kernel_data
class CalibratedPositionParams:
    nominal: SharedArray
    actuator_indices: SharedArray
    noise: np.float32


@dispatch
def calibrated_joint_positions(ctx: ManagerContext, out: np.ndarray, params: CalibratedPositionParams) -> None:
    action: CalibratedActionState = ctx.actions["joint_position"]
    for joint in range(out.size):
        out[joint] = (
            ctx.sim["robot_dof_pos"][joint] - params.nominal[joint] - action.bias[params.actuator_indices[joint]]
        )
    add_uniform_noise(out, params.noise, ctx.rand.state)


@configclass(kw_only=True)
class CalibratedPositionObsCfg(ObservationTermCfg):
    noise: float = 0.0

    def __call__(self, ctx) -> ObsTerm:
        robot = ctx.cfg.scene.objs.robot
        nominal = dict(zip(robot.key_pose.joint_names, robot.key_pose.poses["default"], strict=True))
        joints = ctx.model.bodies["robot"].joint_names
        actuator_joints = tuple(item.target_name for item in ctx.model.actuators)
        params = CalibratedPositionParams(
            nominal=np.asarray([nominal[name] for name in joints], dtype=np.float32),
            actuator_indices=np.asarray([actuator_joints.index(name) for name in joints], dtype=np.int64),
            noise=np.float32(self.noise),
        )
        return ObsTerm(len(joints), calibrated_joint_positions, params)


@njit(inline="always")
def relative_rot6d_columns(reference: np.ndarray, target: np.ndarray, out: np.ndarray) -> None:
    """First two matrix columns, flattened row-major, as in Holosoma PPO."""
    q = np.empty(4, dtype=np.float32)
    inverse(reference, q)
    mul(q, target, q)
    x, y, z, w = q
    s = 2.0 / float(np.dot(q, q))
    out[0] = 1.0 - s * (y * y + z * z)
    out[1] = s * (x * y - z * w)
    out[2] = s * (x * y + z * w)
    out[3] = 1.0 - s * (x * x + z * z)
    out[4] = s * (x * z - y * w)
    out[5] = s * (y * z + x * w)


@dispatch
def reference_rot6d(ctx: ManagerContext, out: np.ndarray, noise: np.float32) -> None:
    motion = ctx.commands["motion"]
    relative_rot6d_columns(
        ctx.sim["tracked_body_quat"][motion.reference_index],
        motion.clip.reference_body_quat_w[motion.steps[0]],
        out,
    )
    add_uniform_noise(out, noise, ctx.rand.state)


@configclass(kw_only=True)
class ReferenceRot6dObsCfg(ObservationTermCfg):
    noise: float = 0.0

    def __call__(self, ctx) -> ObsTerm:
        return ObsTerm(6, reference_rot6d, np.float32(self.noise))


@configclass(kw_only=True)
class ReferencePositionObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        return ObsTerm(3, motion_reference_position_obs)


@dispatch
def body_rot6d(ctx: ManagerContext, out: np.ndarray) -> None:
    motion = ctx.commands["motion"]
    quats = ctx.sim["tracked_body_quat"]
    for index in range(quats.shape[0]):
        relative_rot6d_columns(quats[motion.reference_index], quats[index], out[index * 6 : index * 6 + 6])


@configclass(kw_only=True)
class BodyRot6dObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        return ObsTerm(6 * len(ctx.cfg.commands.motion.tracked_body_names), body_rot6d)


@dispatch
def object_rot6d(ctx: ManagerContext, out: np.ndarray) -> None:
    motion = ctx.commands["motion"]
    relative_rot6d_columns(ctx.sim["tracked_body_quat"][motion.reference_index], ctx.sim["object_quat"], out)


@configclass(kw_only=True)
class ObjectRot6dObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        return ObsTerm(6, object_rot6d)


@dispatch
def body_tracking_failed(
    ctx: ManagerContext, body_indices: tuple[int, ...], height: np.float32, angle: np.float32
) -> bool:
    motion = ctx.commands["motion"]
    ref = motion.reference_index
    step = motion.steps[0]
    z_error = abs(ctx.sim["tracked_body_pos"][ref, 2] - motion.clip.reference_body_pos_w[step, 2])
    for index in body_indices:
        z_error = max(
            z_error, abs(ctx.sim["tracked_body_pos"][index, 2] - motion.target_body_position_relative[index, 2])
        )
    angle_error = rotation_distance(ctx.sim["tracked_body_quat"][ref], motion.clip.reference_body_quat_w[step])
    ctx.metrics["body_height_error"][0] = z_error
    ctx.metrics["body_anchor_angle_error"][0] = angle_error
    return z_error > height or angle_error > angle


@configclass(kw_only=True)
class PaperBodyTerminationCfg(TerminationTermCfg):
    height: float = 0.25
    angle: float = 0.8
    body_names: tuple[str, ...] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )

    def __call__(self, ctx) -> TerminationTerm:
        indices = tuple(ctx.cfg.commands.motion.tracked_body_names.index(name) for name in self.body_names)
        return TerminationTerm(
            body_tracking_failed,
            indices,
            np.float32(self.height),
            np.float32(self.angle),
            metric_names=("body_height_error", "body_anchor_angle_error"),
        )


@kernel_data
class ObjectTerminationGate:
    enabled: SharedArray


@dispatch
def staged_object_failure(
    ctx: ManagerContext, gate: ObjectTerminationGate, position: np.float32, angle: np.float32
) -> bool:
    motion = ctx.commands["motion"]
    step = motion.steps[0]
    delta = ctx.sim["object_pos"] - motion.clip.extensions["object_pos_w"].data[step]
    pos_error = math.sqrt(float(np.dot(delta, delta)))
    ori_error = rotation_distance(ctx.sim["object_quat"], motion.clip.extensions["object_quat_w"].data[step])
    ctx.metrics["object_position_error"][0] = pos_error
    ctx.metrics["object_orientation_error"][0] = ori_error
    ctx.metrics["object_termination_enabled"][0] = 1.0 if gate.enabled[0] else 0.0
    return gate.enabled[0] and (pos_error > position or ori_error > angle)


@configclass(kw_only=True)
class StagedObjectTerminationCfg(TerminationTermCfg):
    position: float = 1.0
    angle: float = math.pi / 4.0

    def __call__(self, ctx) -> TerminationTerm:
        return TerminationTerm(
            staged_object_failure,
            ObjectTerminationGate(np.zeros(1, dtype=np.bool_)),
            np.float32(self.position),
            np.float32(self.angle),
            metric_names=("object_position_error", "object_orientation_error", "object_termination_enabled"),
        )
