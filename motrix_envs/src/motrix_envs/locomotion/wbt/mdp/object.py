# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0

"""Dynamic-object tracking terms sharing the robot WBT reference clock."""

import math

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.manager import (
    ManagerContext,
    ObservationTermCfg,
    ObsTerm,
    ResetTerm,
    ResetTermCfg,
    RewardTerm,
    RewardTermCfg,
    TerminationTerm,
    TerminationTermCfg,
)
from motrix_env_core.manager.math.quaternion import rotate_inverse, rotation_distance
from motrix_env_core.numba.kernel_data import Map, kernel_data
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.sim import BodyAngularVelocityWrite, BodyLinearVelocityWrite, BodyPositionWrite, BodyRotationWrite
from motrix_env_core.sim.write import LinkInertiaWrite, LinkMassWrite
from motrix_envs.locomotion.wbt.mdp.command import WbtMotionCommand
from motrix_envs.locomotion.wbt.mdp.observations import _write_relative_orientation_6d

OBJECT_CHANNELS = ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w")


@dispatch
def reset_object(ctx: ManagerContext, writes: Map[np.ndarray], noise: np.float32) -> None:
    motion: WbtMotionCommand = ctx.commands["motion"]
    step = motion.steps[0]
    writes["position"][0] = motion.clip.extensions["object_pos_w"].data[step]
    writes["rotation"][0] = motion.clip.extensions["object_quat_w"].data[step]
    writes["linear_velocity"][0] = motion.clip.extensions["object_lin_vel_w"].data[step]
    # Holosoma initializes object angular velocity to zero, even when the file
    # contains angular velocity samples.
    writes["angular_velocity"][0] = 0.0
    for axis in range(2):
        writes["position"][0, axis] += ctx.rand.next_uniform() * noise


@configclass(kw_only=True)
class ObjectResetCfg(ResetTermCfg):
    body: str
    noise: float = 0.05
    noise_scale: float = 1.0

    def __call__(self, ctx) -> ResetTerm:
        del ctx
        return ResetTerm(
            reset_object,
            np.float32(self.noise * self.noise_scale),
            writes={
                "position": BodyPositionWrite((self.body,)),
                "rotation": BodyRotationWrite((self.body,)),
                "linear_velocity": BodyLinearVelocityWrite((self.body,)),
                "angular_velocity": BodyAngularVelocityWrite((self.body,)),
            },
        )


@kernel_data
class ObjectMassState:
    """Per-environment startup mass, retained across episode and clip resets."""

    mass: np.ndarray


@dispatch
def reset_object_mass(
    ctx: ManagerContext,
    writes: Map[np.ndarray],
    light_probability: np.float32,
    light_range: np.ndarray,
    heavy_range: np.ndarray,
    inertia_per_mass: np.ndarray,
    state: ObjectMassState,
) -> None:
    env_id = ctx.env_id
    mass = state.mass[env_id, 0]
    if mass == 0.0:
        bounds = heavy_range
        if ctx.rand.uniform_range(0.0, 1.0) < light_probability:
            bounds = light_range
        mass = ctx.rand.uniform_range(bounds[0], bounds[1])
        state.mass[env_id, 0] = mass
    writes["mass"][0] = mass
    for axis in range(6):
        writes["inertia"][0, axis] = inertia_per_mass[axis] * mass


@configclass(kw_only=True)
class ObjectMassRandomizationCfg(ResetTermCfg):
    """Sample mass once per environment; restore it and scaled inertia on reset."""

    body: str
    light_probability: float = 1.0
    light_range: tuple[float, float] = (0.1, 0.1)
    heavy_range: tuple[float, float] = (0.1, 0.1)
    # Original box: mass 0.1 kg, diagonal inertia 0.002 kg m^2.
    inertia_per_mass: tuple[float, ...] = (0.02, 0.02, 0.02, 0.0, 0.0, 0.0)

    def __call__(self, ctx) -> ResetTerm:
        if not 0.0 <= self.light_probability <= 1.0:
            raise ValueError("light_probability must be in [0, 1]")
        for bounds in (self.light_range, self.heavy_range):
            if not 0.0 < bounds[0] <= bounds[1]:
                raise ValueError("object mass ranges must be positive and ordered")
        return ResetTerm(
            reset_object_mass,
            np.float32(self.light_probability),
            np.asarray(self.light_range, dtype=np.float32),
            np.asarray(self.heavy_range, dtype=np.float32),
            np.asarray(self.inertia_per_mass, dtype=np.float32),
            ObjectMassState(mass=np.zeros((ctx.num_envs, 1), dtype=np.float32)),
            writes={"mass": LinkMassWrite((self.body,)), "inertia": LinkInertiaWrite((self.body,))},
        )


@dispatch
def object_position_reward(ctx: ManagerContext, sigma: np.float32) -> float:
    motion: WbtMotionCommand = ctx.commands["motion"]
    error = ctx.sim["object_pos"] - motion.clip.extensions["object_pos_w"].data[motion.steps[0]]
    return math.exp(-float(np.dot(error, error)) / (sigma * sigma))


@dispatch
def object_orientation_reward(ctx: ManagerContext, sigma: np.float32) -> float:
    motion: WbtMotionCommand = ctx.commands["motion"]
    error = rotation_distance(ctx.sim["object_quat"], motion.clip.extensions["object_quat_w"].data[motion.steps[0]])
    return math.exp(-(error * error) / (sigma * sigma))


@configclass(kw_only=True)
class ObjectPositionRewardCfg(RewardTermCfg):
    sigma: float = 0.3

    def __call__(self, ctx) -> RewardTerm:
        del ctx
        return RewardTerm(object_position_reward, np.float32(self.sigma))


@configclass(kw_only=True)
class ObjectOrientationRewardCfg(RewardTermCfg):
    sigma: float = 0.4

    def __call__(self, ctx) -> RewardTerm:
        del ctx
        return RewardTerm(object_orientation_reward, np.float32(self.sigma))


@dispatch
def object_tracking_failed(
    ctx: ManagerContext, position_threshold: np.float32, orientation_threshold: np.float32
) -> bool:
    motion: WbtMotionCommand = ctx.commands["motion"]
    step = motion.steps[0]
    diff = ctx.sim["object_pos"] - motion.clip.extensions["object_pos_w"].data[step]
    position_error = math.sqrt(float(np.dot(diff, diff)))
    orientation_error = rotation_distance(ctx.sim["object_quat"], motion.clip.extensions["object_quat_w"].data[step])
    ctx.metrics["object_position_error"][0] = position_error
    ctx.metrics["object_orientation_error"][0] = orientation_error
    return position_error > position_threshold or orientation_error > orientation_threshold


@configclass(kw_only=True)
class ObjectTrackingTerminationCfg(TerminationTermCfg):
    position_threshold: float = 0.25
    orientation_threshold: float = 0.8

    def __call__(self, ctx) -> TerminationTerm:
        del ctx
        return TerminationTerm(
            object_tracking_failed,
            np.float32(self.position_threshold),
            np.float32(self.orientation_threshold),
            metric_names=("object_position_error", "object_orientation_error"),
        )


@dispatch
def object_position_obs(ctx: ManagerContext, out: np.ndarray) -> None:
    motion: WbtMotionCommand = ctx.commands["motion"]
    index = motion.reference_index
    diff = ctx.sim["object_pos"] - ctx.sim["tracked_body_pos"][index]
    rotate_inverse(ctx.sim["tracked_body_quat"][index], diff, out)


@dispatch
def object_orientation_obs(ctx: ManagerContext, out: np.ndarray) -> None:
    motion: WbtMotionCommand = ctx.commands["motion"]
    _write_relative_orientation_6d(ctx.sim["tracked_body_quat"][motion.reference_index], ctx.sim["object_quat"], out)


@dispatch
def object_linear_velocity_obs(ctx: ManagerContext, out: np.ndarray) -> None:
    motion: WbtMotionCommand = ctx.commands["motion"]
    # Velocity is a vector: rotate it without applying the reference position.
    # Holosoma's obj_lin_vel_b uses subtract_frame_transforms, which also
    # subtracts position; do not reproduce that dimensional inconsistency.
    rotate_inverse(ctx.sim["tracked_body_quat"][motion.reference_index], ctx.sim["object_lin_vel"], out)


@configclass(kw_only=True)
class ObjectPositionObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        del ctx
        return ObsTerm(3, object_position_obs)


@configclass(kw_only=True)
class ObjectOrientationObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        del ctx
        return ObsTerm(6, object_orientation_obs)


@configclass(kw_only=True)
class ObjectLinearVelocityObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        del ctx
        return ObsTerm(3, object_linear_velocity_obs)
