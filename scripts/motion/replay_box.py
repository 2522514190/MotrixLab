# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Kinematically replay a robot and box from a MotrixLab motion NPZ v1."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import motrixsim as mtx
import numpy as np
from absl import app, flags
from motrix_robots.booster import BoosterK1
from motrix_robots.dex_evt import DexEvt
from motrix_robots.unitree import UnitreeG129Dof
from motrixsim.render import RenderApp, RenderClosedError, RenderSettings
from omegaconf import MISSING

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import (
    BodyCfg,
    RobotCfg,
    SystemCameraCfg,
    UrdfFileCfg,
)
from motrix_env_motrixsim.compiler import build_scene_model
from motrix_envs.config.scene import StandardSceneCfg, StandardSceneObjsCfg
from motrix_envs.motion.loader import MotrixMotion
from motrix_envs.motion.tracked import load_extension_channel

logger = logging.getLogger(__name__)

_ROBOT_CFG_FACTORIES: dict[str, Callable[[], RobotCfg]] = {
    "dex-evt": DexEvt,
    "g1-29dof": UnitreeG129Dof,
    "k1": BoosterK1,
}

_ROBOT = flags.DEFINE_enum(
    "robot",
    None,
    sorted(_ROBOT_CFG_FACTORIES),
    "Built-in RobotCfg used to build the replay scene.",
    required=True,
)
_MOTION = flags.DEFINE_string(
    "motion",
    None,
    "MotrixLab v1 motion NPZ file.",
    required=True,
)
_FPS = flags.DEFINE_integer("fps", None, "Override playback frame rate; defaults to the motion's fps.")
_START_STEP = flags.DEFINE_integer("start-step", 0, "Inclusive start frame index.")
_END_STEP = flags.DEFINE_integer("end-step", None, "Exclusive end frame index; defaults to the last frame.")
_LOOP = flags.DEFINE_bool("loop", False, "Loop the clip until the window is closed.")
_SPEED = flags.DEFINE_float("speed", 1.0, "Playback speed multiplier (>0).", lower_bound=0.0)
_OBJECT_MODEL = flags.DEFINE_string(
    "object-model", None, "Box URDF with a baseLink root for reference replay.", required=True
)


@configclass
class ReplaySceneObjsCfg(StandardSceneObjsCfg):
    box: BodyCfg = MISSING


def build_replay_model(robot_cfg: RobotCfg, object_model: str | Path) -> mtx.SceneModel:
    """Build a ground scene with the requested robot and a floating box."""
    scene = StandardSceneCfg(
        system_camera=SystemCameraCfg(distance=6.0, elevation=-20.0, azimuth=180.0),
        objs=ReplaySceneObjsCfg(
            robot=robot_cfg,
            box=BodyCfg(
                model=UrdfFileCfg(file=Path(object_model).expanduser()),
                base_link_name="baseLink",
            ),
        ),
    )
    return build_scene_model(scene)


def _model_joint_indices(model: mtx.SceneModel, motion: MotrixMotion) -> np.ndarray:
    """Resolve motion joint columns into model order and validate free-root layout."""
    joint_names = [str(name) for name in model.joint_names]
    expected_qpos = 7 + len(joint_names) + 7
    expected_qvel = 6 + len(joint_names) + 6
    if model.num_dof_pos != expected_qpos or model.num_dof_vel != expected_qvel:
        raise ValueError(
            "Replay requires a floating-base robot with scalar joints and a single floating box; "
            f"model has qpos/qvel widths {model.num_dof_pos}/{model.num_dof_vel}, "
            f"expected {expected_qpos}/{expected_qvel}."
        )
    return motion.joint_indices(joint_names)


def _root_index(motion: MotrixMotion) -> int:
    """Use explicit root metadata, or the robot pelvis for legacy clips."""
    root_body_name = motion.root_body_name
    if root_body_name is None:
        # Older converted clips include a synthetic ``world`` body first;
        # that body is not the robot floating base.  G1 clips use ``pelvis``
        # as the root body, while retaining the first-body fallback for other
        # legacy robot formats.
        root_body_name = "pelvis" if "pelvis" in motion.body_names else motion.body_names[0]
    return motion.body_index(root_body_name)


def write_frame(
    model: mtx.SceneModel,
    data: mtx.SceneData,
    motion: MotrixMotion,
    joint_indices: np.ndarray,
    root_index: int,
    step: int,
    box_motion: dict[str, np.ndarray],
) -> None:
    """Write one motion frame into SceneData and run forward kinematics."""
    qpos = np.zeros((1, model.num_dof_pos), dtype=np.float32)
    qvel = np.zeros((1, model.num_dof_vel), dtype=np.float32)

    qpos[0, :3] = motion.body_pos_w[step, root_index]
    qpos[0, 3:7] = motion.body_quat_w[step, root_index]
    robot_qpos_end = 7 + len(joint_indices)
    qpos[0, 7:robot_qpos_end] = motion.joint_pos[step, joint_indices]

    qvel[0, :3] = motion.body_lin_vel_w[step, root_index]
    qvel[0, 3:6] = motion.body_ang_vel_w[step, root_index]
    robot_qvel_end = 6 + len(joint_indices)
    qvel[0, 6:robot_qvel_end] = motion.joint_vel[step, joint_indices]

    # Fill both floating bases before submitting: set_dof_pos validates all
    # quaternions, including the object's, immediately.
    qpos[0, -7:-4] = box_motion["object_pos_w"][step]
    qpos[0, -4:] = box_motion["object_quat_w"][step]
    qvel[0, -6:-3] = box_motion["object_lin_vel_w"][step]
    qvel[0, -3:] = box_motion["object_ang_vel_w"][step]
    data.set_dof_pos(qpos, model)
    data.set_dof_vel(qvel)
    model.forward_kinematic(data)


def _frame_range(motion: MotrixMotion, start: int, end: int | None) -> tuple[int, int]:
    total = motion.num_frames
    start = max(0, min(start, total - 1))
    end = total if end is None else max(start + 1, min(end, total))
    return start, end


def _launch_renderer(model: mtx.SceneModel) -> RenderApp:
    settings = RenderSettings.performance()
    settings.enable_shadow = True
    renderer = RenderApp()
    renderer.launch(
        model,
        batch=1,
        render_offset=[[0.0, 0.0, 0.0]],
        render_settings=settings,
    )
    renderer.system_camera.set_view([0.0, 0.0, 0.75], 6.0, -20.0, 180.0)
    renderer.system_camera.active = True
    return renderer


def replay(
    robot_cfg: RobotCfg,
    motion_path: str | Path,
    object_model: str | Path,
    *,
    fps: int | None = None,
    start_step: int = 0,
    end_step: int | None = None,
    loop: bool = False,
    speed: float = 1.0,
) -> None:
    """Replay the robot and box states together, without integrating physics."""
    motion = MotrixMotion(motion_path)
    box_motion = {}
    for name, width in (("object_pos_w", 3), ("object_quat_w", 4), ("object_lin_vel_w", 3), ("object_ang_vel_w", 3)):
        channel = load_extension_channel(motion, name)
        if channel.shape != (motion.num_frames, width):
            raise ValueError(f"Expected ext_{name} to have shape ({motion.num_frames}, {width}), got {channel.shape}.")
        box_motion[name] = channel
    model = build_replay_model(robot_cfg, object_model)
    joint_indices = _model_joint_indices(model, motion)
    root_index = _root_index(motion)
    start, end = _frame_range(motion, start_step, end_step)

    playback_fps = motion.fps if fps is None else fps
    if playback_fps <= 0:
        raise ValueError(f"fps must be positive, got {playback_fps}")
    if speed <= 0.0:
        raise ValueError(f"speed must be positive, got {speed}")
    frame_dt = 1.0 / (playback_fps * speed)

    logger.info(
        "Replaying %s with %s: frames [%d, %d) of %d @ %d fps (speed=%.2fx)",
        motion.path,
        type(robot_cfg).__name__,
        start,
        end,
        motion.num_frames,
        playback_fps,
        speed,
    )

    data = mtx.SceneData(model, batch=[1])
    data.reset(model)
    renderer = _launch_renderer(model)
    steps = range(start, end)
    try:
        while True:
            for step in steps:
                t0 = time.monotonic()
                write_frame(model, data, motion, joint_indices, root_index, step, box_motion)
                renderer.sync(data=data)
                sleep_dt = frame_dt - (time.monotonic() - t0)
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
            if not loop:
                break
    except RenderClosedError:
        logger.info("Render window closed.")
    finally:
        renderer.__exit__(None, None, None)


def main(argv):
    del argv  # unused
    replay(
        _ROBOT_CFG_FACTORIES[_ROBOT.value](),
        _MOTION.value,
        _OBJECT_MODEL.value,
        fps=_FPS.value,
        start_step=_START_STEP.value,
        end_step=_END_STEP.value,
        loop=_LOOP.value,
        speed=_SPEED.value,
    )


if __name__ == "__main__":
    app.run(main)
