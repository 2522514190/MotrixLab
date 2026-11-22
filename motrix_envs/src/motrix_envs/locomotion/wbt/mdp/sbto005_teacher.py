# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Batched object observations, reset and rewards for the SBTO 005 teacher."""

import torch

from motrix_env_core.config import configclass
from motrix_env_core.config.sim_reset import ResetTermCfg
from motrix_env_core.manager.context import BuildContext, ManagerContext
from motrix_env_core.manager.kernel_data import Map
from motrix_env_core.manager.observations import ObservationTermCfg, ObsTerm
from motrix_env_core.manager.rewards import RewardTerm, RewardTermCfg
from motrix_env_core.manager.sim_reset import ResetTerm
from motrix_env_core.math.quaternion import inverse, matrix6d, mul, rotate_inverse, rotation_distance
from motrix_env_core.mdp.observations import add_uniform_noise
from motrix_env_core.sim.write import (
    BodyAngularVelocityWrite,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
)

OBJECT_CHANNELS = ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w")
CONTACT_CHANNEL = "contact_computed"
CONTACT_NAMES = ("obj_static_cnt", "left_hand_cnt", "right_hand_cnt")
HAND_LINKS = ("left_wrist_pitch_link", "left_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_yaw_link")


def reset_box(ctx: ManagerContext, env_ids: torch.Tensor, writes: Map[torch.Tensor], initial_xy_noise: float) -> None:
    motion = ctx.commands["motion"]
    frames = motion.steps[env_ids, 0]
    extensions = motion.clip.extensions
    position = writes["position"][:, 0]
    position.copy_(extensions["object_pos_w"].data[frames])
    if initial_xy_noise:
        offset = (2 * ctx.rand.uniform((env_ids.numel(), 2), device=env_ids.device) - 1) * initial_xy_noise
        # Later reference starts must retain the box pose paired with the robot.
        position[:, :2] += offset * (frames == 0)[:, None]
    writes["rotation"][:, 0] = extensions["object_quat_w"].data[frames]
    writes["linear_velocity"][:, 0] = extensions["object_lin_vel_w"].data[frames]
    # The reference reset starts with zero angular velocity.
    writes["angular_velocity"].zero_()


@configclass(kw_only=True)
class BoxResetCfg(ResetTermCfg):
    body: str
    initial_xy_noise: float = 0.0

    def __call__(self, ctx: BuildContext) -> ResetTerm:
        return ResetTerm(
            reset_box,
            self.initial_xy_noise,
            writes={
                "position": BodyPositionWrite((self.body,)),
                "rotation": BodyRotationWrite((self.body,)),
                "linear_velocity": BodyLinearVelocityWrite((self.body,)),
                "angular_velocity": BodyAngularVelocityWrite((self.body,)),
            },
        )


def box_pose_obs(ctx: ManagerContext, out: torch.Tensor, error: bool, noise: float) -> None:
    motion = ctx.commands["motion"]
    anchor = motion.reference_index
    torso_quat = ctx.sim["tracked_body_quat"][:, anchor]
    torso_inverse = inverse(torso_quat)
    actual_quat_b = mul(torso_inverse, ctx.sim["object_quat"])
    actual_pos = ctx.sim["object_pos"]
    if error:
        frames = motion.steps[:, 0]
        desired_pos = motion.clip.extensions["object_pos_w"].data[frames]
        desired_quat = motion.clip.extensions["object_quat_w"].data[frames]
        displacement = desired_pos - actual_pos
        rotation = mul(mul(torso_inverse, desired_quat), inverse(actual_quat_b))
    else:
        displacement = actual_pos - ctx.sim["tracked_body_pos"][:, anchor]
        rotation = actual_quat_b
    out[:, :3] = rotate_inverse(torso_quat, displacement)
    out[:, 3:] = matrix6d(rotation)
    add_uniform_noise(ctx, out, noise)


@configclass(kw_only=True)
class BoxPoseObsCfg(ObservationTermCfg):
    error: bool = False
    noise: float = 0.0

    def __call__(self, ctx: BuildContext) -> ObsTerm:
        return ObsTerm(9, box_pose_obs, self.error, self.noise)


def box_velocity_error_obs(ctx: ManagerContext, out: torch.Tensor) -> None:
    motion = ctx.commands["motion"]
    frames = motion.steps[:, 0]
    out[:, :3] = motion.clip.extensions["object_lin_vel_w"].data[frames] - ctx.sim["object_lin_vel"]
    out[:, 3:] = motion.clip.extensions["object_ang_vel_w"].data[frames] - ctx.sim["object_ang_vel"]


@configclass(kw_only=True)
class BoxVelocityErrorObsCfg(ObservationTermCfg):
    def __call__(self, ctx: BuildContext) -> ObsTerm:
        return ObsTerm(6, box_velocity_error_obs)


def reference_body_orientation_obs(ctx: ManagerContext, out: torch.Tensor) -> None:
    quats = ctx.sim["tracked_body_quat"]
    anchor = ctx.commands["motion"].reference_index
    local_quats = mul(inverse(quats[:, anchor : anchor + 1]), quats)
    out.copy_(matrix6d(local_quats).flatten(start_dim=1))


@configclass(kw_only=True)
class ReferenceBodyOrientationObsCfg(ObservationTermCfg):
    def __call__(self, ctx: BuildContext) -> ObsTerm:
        return ObsTerm(6 * len(ctx.cfg.commands.motion.tracked_body_names), reference_body_orientation_obs)


def box_position_reward(ctx: ManagerContext, sigma: float) -> torch.Tensor:
    motion = ctx.commands["motion"]
    delta = motion.clip.extensions["object_pos_w"].data[motion.steps[:, 0]] - ctx.sim["object_pos"]
    return torch.exp(-torch.linalg.vector_norm(delta, dim=-1) / sigma)


def box_orientation_reward(ctx: ManagerContext, sigma: float) -> torch.Tensor:
    motion = ctx.commands["motion"]
    ref = motion.clip.extensions["object_quat_w"].data[motion.steps[:, 0]]
    return torch.exp(-rotation_distance(ref, ctx.sim["object_quat"]) / sigma)


def box_velocity_reward(ctx: ManagerContext, sigma: float, channel: str, query: str) -> torch.Tensor:
    motion = ctx.commands["motion"]
    delta = motion.clip.extensions[channel].data[motion.steps[:, 0]] - ctx.sim[query]
    return torch.exp(-delta.square().sum(dim=-1) / (sigma * sigma))


@configclass(kw_only=True)
class BoxRewardCfg(RewardTermCfg):
    quantity: str
    sigma: float

    def __call__(self, ctx: BuildContext) -> RewardTerm:
        if self.quantity == "position":
            return RewardTerm(box_position_reward, self.sigma)
        if self.quantity == "orientation":
            return RewardTerm(box_orientation_reward, self.sigma)
        query = {"linear_velocity": "object_lin_vel", "angular_velocity": "object_ang_vel"}[self.quantity]
        return RewardTerm(box_velocity_reward, self.sigma, f"{query}_w", query)


def hand_contact_reward(ctx: ManagerContext, hand: int) -> torch.Tensor:
    motion = ctx.commands["motion"]
    reference = motion.clip.extensions[CONTACT_CHANNEL].data[motion.steps[:, 0], hand + 1] > 0
    force = ctx.sim["hand_box_contact_forces"].reshape(-1, 2, 3)[:, hand]
    in_contact = torch.linalg.vector_norm(force, dim=-1) > 1.0
    return (in_contact & reference).to(dtype=torch.float32)


def hand_grasp_reward(ctx: ManagerContext, hand: int, box_height: float) -> torch.Tensor:
    motion = ctx.commands["motion"]
    reference = motion.clip.extensions[CONTACT_CHANNEL].data[motion.steps[:, 0], hand + 1] > 0
    report = ctx.sim["hand_box_contact_points"].reshape(reference.shape[0], 2, -1)[:, hand]
    # MotrixSim reports one contact count followed by fixed slots of [normal, tangent1, tangent2, x, y, z].
    points = report[:, 1:].reshape(reference.shape[0], -1, 6)
    valid = torch.arange(points.shape[1], device=points.device) < report[:, :1]
    normal_force = torch.where(valid, points[..., 0].clamp_min(0.0), 0.0)
    total_force = normal_force.sum(dim=-1)

    offset_w = points[..., 3:6] - ctx.sim["object_pos"][:, None]
    local_z = rotate_inverse(ctx.sim["object_quat"][:, None], offset_w)[..., 2]
    height = local_z / box_height + 0.5
    height_score = ((0.95 - height) / 0.15).clamp(0.0, 1.0)
    height_score = torch.where(height >= 0.35 - 1e-6, height_score, 0.0)
    hand_height_score = (height_score * normal_force).sum(dim=-1) / total_force.clamp_min(1e-6)

    force_score = ((total_force - 2.0) / 13.0).clamp(0.0, 1.0)
    force_score *= torch.exp(-0.1 * (total_force - 20.0).clamp_min(0.0))
    return torch.where(reference, hand_height_score * force_score, 0.0)


@configclass(kw_only=True)
class HandContactRewardCfg(RewardTermCfg):
    hand: int
    box_height: float | None = None

    def __call__(self, ctx: BuildContext) -> RewardTerm:
        if self.box_height is not None:
            return RewardTerm(hand_grasp_reward, self.hand, self.box_height)
        return RewardTerm(hand_contact_reward, self.hand)


def joint_position_reward(ctx: ManagerContext, sigma: float) -> torch.Tensor:
    motion = ctx.commands["motion"]
    delta = motion.clip.joint_pos[motion.steps[:, 0]] - ctx.sim["robot_dof_pos"]
    return torch.exp(-delta.square().mean(dim=-1) / (sigma * sigma))


@configclass(kw_only=True)
class JointPositionRewardCfg(RewardTermCfg):
    sigma: float

    def __call__(self, ctx: BuildContext) -> RewardTerm:
        return RewardTerm(joint_position_reward, self.sigma)


def joint_acceleration_squared(ctx: ManagerContext, pre_step_velocity: torch.Tensor) -> torch.Tensor:
    delta = ctx.sim["robot_dof_vel"] - pre_step_velocity
    return delta.square().sum(dim=-1) / (ctx.dt * ctx.dt)


@configclass(kw_only=True)
class JointAccelerationSquaredCfg(RewardTermCfg):
    def __call__(self, ctx: BuildContext) -> RewardTerm:
        num_joints = len(ctx.model.bodies["robot"].joint_names)
        return RewardTerm(joint_acceleration_squared, torch.zeros(ctx.num_envs, num_joints))


def joint_torque_squared(ctx: ManagerContext, pd_effort: torch.Tensor) -> torch.Tensor:
    return pd_effort


@configclass(kw_only=True)
class JointTorqueSquaredCfg(RewardTermCfg):
    def __call__(self, ctx: BuildContext) -> RewardTerm:
        return RewardTerm(joint_torque_squared, torch.zeros(ctx.num_envs))
