# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Object observations and contact rewards for the SBTO 005 teacher."""

import math

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerContext, ObservationTermCfg, ObsTerm, RewardTerm, RewardTermCfg
from motrix_env_core.manager.math.quaternion import (
    inverse,
    mul,
    rotate_inverse,
    rotation_distance,
    to_matrix_first_two_rows,
)
from motrix_env_core.mdp.noise import add_uniform_noise
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_envs.locomotion.wbt.mdp.command import WbtMotionCommand

CONTACT_CHANNEL = "contact_computed"
CONTACT_NAMES = ("obj_static_cnt", "left_hand_cnt", "right_hand_cnt")
HAND_LINKS = ("left_wrist_pitch_link", "left_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_yaw_link")


@dispatch
def box_pose_obs(ctx: ManagerContext, out: np.ndarray, error: bool, noise: np.float32) -> None:
    motion: WbtMotionCommand = ctx.commands["motion"]
    anchor = motion.reference_index
    torso_quat = ctx.sim["tracked_body_quat"][anchor]
    torso_inverse = np.empty(4, dtype=np.float32)
    actual_quat_b = np.empty(4, dtype=np.float32)
    inverse(torso_quat, torso_inverse)
    mul(torso_inverse, ctx.sim["object_quat"], actual_quat_b)
    actual_pos = ctx.sim["object_pos"]
    if error:
        frame = motion.steps[0]
        desired_pos = motion.clip.extensions["object_pos_w"].data[frame]
        desired_quat = motion.clip.extensions["object_quat_w"].data[frame]
        rotate_inverse(torso_quat, desired_pos - actual_pos, out[:3])
        # Express desired * inverse(actual) in the current torso frame.
        inverse(actual_quat_b, actual_quat_b)
        mul(torso_inverse, desired_quat, torso_inverse)
        mul(torso_inverse, actual_quat_b, actual_quat_b)
    else:
        rotate_inverse(torso_quat, actual_pos - ctx.sim["tracked_body_pos"][anchor], out[:3])
    to_matrix_first_two_rows(actual_quat_b, out[3:])
    if noise > 0.0:
        add_uniform_noise(out, noise, ctx.rand.state)


@configclass(kw_only=True)
class BoxPoseObsCfg(ObservationTermCfg):
    error: bool = False
    noise: float = 0.0

    def __call__(self, ctx) -> ObsTerm:
        return ObsTerm(9, box_pose_obs, self.error, np.float32(self.noise))


@dispatch
def box_velocity_error_obs(ctx: ManagerContext, out: np.ndarray) -> None:
    motion: WbtMotionCommand = ctx.commands["motion"]
    frame = motion.steps[0]
    out[:3] = motion.clip.extensions["object_lin_vel_w"].data[frame] - ctx.sim["object_lin_vel"]
    out[3:] = motion.clip.extensions["object_ang_vel_w"].data[frame] - ctx.sim["object_ang_vel"]


@configclass(kw_only=True)
class BoxVelocityErrorObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        return ObsTerm(6, box_velocity_error_obs)


@dispatch
def box_position_reward(ctx: ManagerContext, sigma: np.float32) -> float:
    motion: WbtMotionCommand = ctx.commands["motion"]
    delta = motion.clip.extensions["object_pos_w"].data[motion.steps[0]] - ctx.sim["object_pos"]
    return math.exp(-math.sqrt(float(np.sum(delta * delta))) / sigma)


@dispatch
def box_orientation_reward(ctx: ManagerContext, sigma: np.float32) -> float:
    motion: WbtMotionCommand = ctx.commands["motion"]
    ref = motion.clip.extensions["object_quat_w"].data[motion.steps[0]]
    return math.exp(-rotation_distance(ref, ctx.sim["object_quat"]) / sigma)


@dispatch
def box_linear_velocity_reward(ctx: ManagerContext, sigma: np.float32) -> float:
    motion: WbtMotionCommand = ctx.commands["motion"]
    delta = motion.clip.extensions["object_lin_vel_w"].data[motion.steps[0]] - ctx.sim["object_lin_vel"]
    return math.exp(-float(np.sum(delta * delta)) / (sigma * sigma))


@dispatch
def box_angular_velocity_reward(ctx: ManagerContext, sigma: np.float32) -> float:
    motion: WbtMotionCommand = ctx.commands["motion"]
    delta = motion.clip.extensions["object_ang_vel_w"].data[motion.steps[0]] - ctx.sim["object_ang_vel"]
    return math.exp(-float(np.sum(delta * delta)) / (sigma * sigma))


@configclass(kw_only=True)
class BoxRewardCfg(RewardTermCfg):
    quantity: str
    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        reward = {
            "position": box_position_reward,
            "orientation": box_orientation_reward,
            "linear_velocity": box_linear_velocity_reward,
            "angular_velocity": box_angular_velocity_reward,
        }[self.quantity]
        return RewardTerm(reward, np.float32(self.sigma))


@dispatch
def hand_contact_reward(ctx: ManagerContext, hand: int) -> float:
    motion: WbtMotionCommand = ctx.commands["motion"]
    reference = motion.clip.extensions["contact_computed"].data[motion.steps[0], hand + 1] > 0.0
    forces = ctx.sim["hand_contact_forces"]
    peak_force_squared = 0.0
    for index in range(2 * hand, 2 * hand + 2):
        fx, fy, fz = forces[index, :3]
        peak_force_squared = max(peak_force_squared, fx * fx + fy * fy + fz * fz)
    if (peak_force_squared > 1.0) != reference:
        return 0.0
    return math.exp(-0.1 * max(math.sqrt(peak_force_squared) - 10.0, 0.0))


@configclass(kw_only=True)
class HandContactRewardCfg(RewardTermCfg):
    hand: int

    def __call__(self, ctx) -> RewardTerm:
        return RewardTerm(hand_contact_reward, self.hand)


@dispatch
def joint_acceleration_squared(ctx: ManagerContext, pre_step_velocity: np.ndarray) -> float:
    delta = ctx.sim["robot_dof_vel"] - pre_step_velocity[ctx.env_id]
    return float(np.sum(delta * delta)) / (ctx.dt * ctx.dt)


@configclass(kw_only=True)
class JointAccelerationSquaredCfg(RewardTermCfg):
    def __call__(self, ctx) -> RewardTerm:
        shape = (ctx.num_envs, ctx.num_actuators)
        return RewardTerm(joint_acceleration_squared, np.zeros(shape, dtype=np.float32))


@dispatch
def joint_torque_squared(ctx: ManagerContext, pd_effort: np.ndarray) -> float:
    return float(pd_effort[ctx.env_id])


@configclass(kw_only=True)
class JointTorqueSquaredCfg(RewardTermCfg):
    def __call__(self, ctx) -> RewardTerm:
        return RewardTerm(joint_torque_squared, np.zeros(ctx.num_envs, dtype=np.float32))
