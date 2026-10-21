# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0

"""Raw tracking diagnostics, evaluated before reset without terminating episodes."""

import math

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerContext, TerminationTerm, TerminationTermCfg
from motrix_env_core.manager.math.quaternion import rotation_distance
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_envs.locomotion.wbt.mdp.command import WbtMotionCommand


@dispatch
def largebox_tracking_metrics(ctx: ManagerContext) -> bool:
    motion: WbtMotionCommand = ctx.commands["motion"]
    step = motion.steps[0]
    positions = ctx.sim["tracked_body_pos"]
    orientations = ctx.sim["tracked_body_quat"]
    count = positions.shape[0]

    diff = motion.clip.reference_body_pos_w[step] - positions[motion.reference_index]
    ctx.metrics["tracking/reference_position_error_m"][0] = math.sqrt(float(np.sum(diff * diff)))
    ctx.metrics["tracking/reference_orientation_error_rad"][0] = rotation_distance(
        motion.clip.reference_body_quat_w[step], orientations[motion.reference_index]
    )
    # RMS of body-vector distances, matching the mean squared errors used by rewards.
    diff = motion.target_body_position_relative - positions
    ctx.metrics["tracking/body_relative_position_rms_m"][0] = math.sqrt(float(np.sum(diff * diff)) / count)
    angle_sq = 0.0
    for body_id in range(count):
        angle = rotation_distance(motion.target_body_orientation_relative[body_id], orientations[body_id])
        angle_sq += angle * angle
    ctx.metrics["tracking/body_relative_orientation_rms_rad"][0] = math.sqrt(angle_sq / count)
    diff = motion.clip.tracked_bodies_lin_vel_w[step] - ctx.sim["tracked_body_linear_velocity"]
    ctx.metrics["tracking/body_linear_velocity_rms_m_s"][0] = math.sqrt(float(np.sum(diff * diff)) / count)
    diff = motion.clip.tracked_bodies_ang_vel_w[step] - ctx.sim["tracked_body_angular_velocity"]
    ctx.metrics["tracking/body_angular_velocity_rms_rad_s"][0] = math.sqrt(float(np.sum(diff * diff)) / count)

    diff = motion.clip.joint_pos[step] - ctx.sim["robot_dof_pos"]
    ctx.metrics["tracking/joint_position_rms_rad"][0] = math.sqrt(float(np.mean(diff * diff)))
    diff = motion.clip.joint_vel[step] - ctx.sim["robot_dof_vel"]
    ctx.metrics["tracking/joint_velocity_rms_rad_s"][0] = math.sqrt(float(np.mean(diff * diff)))

    diff = motion.clip.extensions["object_lin_vel_w"].data[step] - ctx.sim["object_lin_vel"]
    ctx.metrics["tracking/object_linear_velocity_error_m_s"][0] = math.sqrt(float(np.sum(diff * diff)))
    diff = motion.clip.extensions["object_ang_vel_w"].data[step] - ctx.sim["object_ang_vel"]
    ctx.metrics["tracking/object_angular_velocity_error_rad_s"][0] = math.sqrt(float(np.sum(diff * diff)))
    return False


@configclass(kw_only=True)
class LargeBoxTrackingMetricsCfg(TerminationTermCfg):
    """Use the pre-reset termination pass to log errors; always return False.

    Box position/orientation errors are already exported by object_tracking.
    """

    def __call__(self, ctx) -> TerminationTerm:
        del ctx
        return TerminationTerm(
            largebox_tracking_metrics,
            metric_names=(
                "tracking/reference_position_error_m",
                "tracking/reference_orientation_error_rad",
                "tracking/body_relative_position_rms_m",
                "tracking/body_relative_orientation_rms_rad",
                "tracking/body_linear_velocity_rms_m_s",
                "tracking/body_angular_velocity_rms_rad_s",
                "tracking/joint_position_rms_rad",
                "tracking/joint_velocity_rms_rad_s",
                "tracking/object_linear_velocity_error_m_s",
                "tracking/object_angular_velocity_error_rad_s",
            ),
        )
