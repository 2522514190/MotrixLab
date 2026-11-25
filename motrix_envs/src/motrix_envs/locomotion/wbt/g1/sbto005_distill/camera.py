# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Left-IR depth contract for the SBTO005 student."""

from math import cos, sin
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import torch

from motrix_env_core.config import configclass
from motrix_env_core.config.scene.sensor import PinholeCameraCfg, RaycastCameraSensorCfg
from motrix_env_core.config.scene.site import SiteCfg
from motrix_env_core.sim.read import RaycastCameraImageQuery
from motrix_envs.locomotion.wbt.g1.sbto005_teacher import Sbto005TeacherCfg, TeacherSceneSensorsCfg

WIDTH, HEIGHT = 64, 48
NEAR, FAR = 0.168, 2.5
SITE = "student_left_ir"
DEPTH = "student_depth"
# RealSense 848x480 left IR, integer pixel centers, measured on our G1.
SOURCE_FOCAL = (431.6185913, 431.6185913)
SOURCE_CENTER = (428.3400574, 242.8603973)
CROP_X = 104
DOWNSAMPLE = 10
# Unitree G1 rev 1.0 d435_joint, expressed in the torso_link body frame.
MOUNT_POSITION = (0.0576235, 0.01753, 0.42987)
MOUNT_PITCH = 0.8307767239493009
HEAD_GROUP = 7


def projection() -> PinholeCameraCfg:
    fx, fy = SOURCE_FOCAL
    cx, cy = SOURCE_CENTER
    return PinholeCameraCfg(
        fx=fx / DOWNSAMPLE,
        fy=fy / DOWNSAMPLE,
        cx=(cx + 0.5 - CROP_X) / DOWNSAMPLE,
        cy=(cy + 0.5) / DOWNSAMPLE,
    )


def optical_rotation() -> tuple[float, float, float, float]:
    # R_y(pitch) * R_body_from_optical. Optical right/down/forward map to
    # body -Y/-Z/+X before applying the pitch. Site quaternions are xyzw.
    c, s = cos(MOUNT_PITCH / 2), sin(MOUNT_PITCH / 2)
    return (-(c + s) / 2, (c + s) / 2, (s - c) / 2, (c - s) / 2)


def camera_sensor(*, site: str = SITE, near: float = NEAR, filtered: bool = True) -> RaycastCameraSensorCfg:
    return RaycastCameraSensorCfg(
        site=site,
        width=WIDTH,
        height=HEIGHT,
        projection=projection(),
        near=near,
        far=FAR,
        update_period=0.02,
        exclude="none",
        include_geom_groups=[group for group in range(8) if group != HEAD_GROUP] if filtered else None,
    )


@configclass(kw_only=True)
class DepthSceneSensorsCfg(TeacherSceneSensorsCfg):
    student_depth: RaycastCameraSensorCfg


def isolate_head_for_rays(source: Path, destination: Path) -> None:
    """Make a task-local MJCF with a separate visibility group for the head."""
    model = ElementTree.parse(source)
    compiler = model.getroot().find("compiler")
    compiler.set("meshdir", str((source.parent / compiler.get("meshdir", ".")).resolve()))
    head = [
        geom for geom in model.iter("geom") if geom.get("name") == "head_collision" or geom.get("mesh") == "head_link"
    ]
    if len(head) != 2 or any(geom.get("group") == str(HEAD_GROUP) for geom in model.iter("geom")):
        raise ValueError(f"Unexpected head mesh or geometry groups in {source}")
    for geom in head:
        geom.set("group", str(HEAD_GROUP))
    model.write(destination, encoding="unicode")


def attach_depth_camera(cfg: Sbto005TeacherCfg, *, robot_mjcf: Path | None = None, near: float = NEAR) -> None:
    """Add depth to a private Teacher config, optionally using the head-group MJCF."""
    robot = cfg.scene.objs.robot
    robot.model.sites.append(
        SiteCfg(name=SITE, parent_link_name="torso_link", position=MOUNT_POSITION, rotation=optical_rotation())
    )
    if robot_mjcf is not None:
        robot.model.file = robot_mjcf
    cfg.scene.sensors = DepthSceneSensorsCfg(
        student_depth=camera_sensor(site=robot.resolve_name(SITE), near=near, filtered=robot_mjcf is not None),
        **dict(cfg.scene.sensors.items()),
    )
    cfg.queries.data[DEPTH] = RaycastCameraImageQuery(sensor=DEPTH)


def aligned_ray_pixels() -> tuple[np.ndarray, np.ndarray]:
    """Return source pixel indices for the 64x48 deployment pinhole."""
    p = projection()
    x = (np.arange(WIDTH) + 0.5 - p.cx) * SOURCE_FOCAL[0] / p.fx + SOURCE_CENTER[0]
    y = (np.arange(HEIGHT) + 0.5 - p.cy) * SOURCE_FOCAL[1] / p.fy + SOURCE_CENTER[1]
    # Use the higher index for the exact half-pixel ties in this 10x reduction.
    return np.floor(x + 0.5 + 1e-9).astype(np.intp), np.floor(y + 0.5 + 1e-9).astype(np.intp)


def resample_left_ir_depth(depth: np.ndarray) -> np.ndarray:
    """Sample metric depth from a rectified 848x480 left-IR image."""
    if depth.shape[-2:] != (480, 848):
        raise ValueError("Expected rectified 848x480 left-IR depth.")
    x, y = aligned_ray_pixels()
    return depth[..., y[:, None], x]


def student_depth(depth: torch.Tensor) -> torch.Tensor:
    """Convert batched meter depth to float32 NCHW without modifying the simulator buffer."""
    meters = depth.to(torch.float32)
    valid = torch.isfinite(meters) & (meters > 0)
    return torch.where(valid, meters, FAR).clamp_(NEAR, FAR).unsqueeze(1).div_(FAR)
