# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0

"""Tracking diagnostics retain physical units and never terminate an episode."""

from types import SimpleNamespace

import numpy as np
import pytest

from motrix_envs.locomotion.wbt.mdp.tracking_metrics import LargeBoxTrackingMetricsCfg, largebox_tracking_metrics


def test_tracking_errors_measure_distances_and_rms_before_reset():
    quat = np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32)
    zeros = np.zeros((2, 3), dtype=np.float32)
    cfg = LargeBoxTrackingMetricsCfg()
    ctx = SimpleNamespace(
        commands={
            "motion": SimpleNamespace(
                steps=np.array([0]),
                reference_index=0,
                target_body_position_relative=zeros.copy(),
                target_body_orientation_relative=quat.copy(),
                clip=SimpleNamespace(
                    reference_body_pos_w=np.zeros((1, 3)),
                    reference_body_quat_w=quat[:1],
                    tracked_bodies_lin_vel_w=zeros[None],
                    tracked_bodies_ang_vel_w=zeros[None],
                    joint_pos=np.zeros((1, 2)),
                    joint_vel=np.zeros((1, 2)),
                    extensions={
                        "object_lin_vel_w": SimpleNamespace(data=np.zeros((1, 3))),
                        "object_ang_vel_w": SimpleNamespace(data=np.zeros((1, 3))),
                    },
                ),
            )
        },
        sim={
            "tracked_body_pos": zeros.copy(),
            "tracked_body_quat": quat.copy(),
            "tracked_body_linear_velocity": zeros.copy(),
            "tracked_body_angular_velocity": zeros.copy(),
            "robot_dof_pos": np.zeros(2),
            "robot_dof_vel": np.zeros(2),
            "object_lin_vel": np.zeros(3),
            "object_ang_vel": np.zeros(3),
        },
        metrics={name: np.zeros(1) for name in cfg(None).metric_names},
    )
    assert not largebox_tracking_metrics(ctx)
    assert all(value[0] == pytest.approx(0) for value in ctx.metrics.values())
    ctx.sim["tracked_body_pos"][0] = [3, 4, 0]
    ctx.sim["tracked_body_quat"][0] = [0, 0, 1, 0]
    ctx.sim["tracked_body_linear_velocity"][0] = [3, 4, 0]
    ctx.sim["tracked_body_angular_velocity"][0] = [0, 0, 2]
    ctx.sim["robot_dof_pos"][:] = [3, 4]
    ctx.sim["robot_dof_vel"][:] = [0, 2]
    ctx.sim["object_lin_vel"][:] = [3, 4, 0]
    ctx.sim["object_ang_vel"][:] = [0, 0, 2]
    assert not largebox_tracking_metrics(ctx)
    expected = (
        5,
        np.pi,
        5 / np.sqrt(2),
        np.pi / np.sqrt(2),
        5 / np.sqrt(2),
        np.sqrt(2),
        5 / np.sqrt(2),
        np.sqrt(2),
        5,
        2,
    )
    for name, value in zip(cfg(None).metric_names, expected, strict=True):
        assert ctx.metrics[name][0] == pytest.approx(value)
