# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared G1 whole-body tracking configuration."""

from dataclasses import InitVar
from pathlib import Path

from motrix_robots.unitree import UnitreeG129Dof

from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import BodyCfg, SystemCameraCfg, UrdfFileCfg
from motrix_env_core.mdp.rewards import ActionRateRewardCfg
from motrix_env_core.sim import BodyLinkNetContactForceQuery
from motrix_envs.config.scene import StandardSceneCfg, StandardSceneObjsCfg
from motrix_envs.locomotion.wbt.cfg import CommandsCfg, RewardsCfg, TerminationsCfg, WbtEnvCfg
from motrix_envs.locomotion.wbt.mdp.command import WbtMotionCommandCfg
from motrix_envs.locomotion.wbt.mdp.terminations import BadBodyZTerminationCfg


@configclass
class G1WbtSceneObjsCfg(StandardSceneObjsCfg):
    """G1 WBT scene with an optional dynamic large box."""

    box: BodyCfg | None = None


def make_large_box() -> BodyCfg:
    """Create the dynamic Holosoma large-box asset for the G1 scene."""
    return BodyCfg(
        model=UrdfFileCfg(file=LARGE_BOX_DIR / "large_box.urdf"),
        base_link_name="baseLink",
        # First frame of Holosoma's object-interaction reference clip.  The
        # mesh is centered near its inertial origin and rests at approximately
        # z=0.18 m on the ground in this pose.
        translation=(0.6762383, 0.49337167, 0.18436243),
        rotation=(0.94689578, 0.31156415, 0.03932899, -0.06906096),
    )


MOTION_DIR = Path(__file__).resolve().parent.parent / "assets" / "motion" / "g1"
LARGE_BOX_DIR = Path(__file__).resolve().parent.parent / "assets" / "objects" / "largebox"
G1_TRACKED_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


@configclass(kw_only=True)
class G1WbtEnvCfg(WbtEnvCfg):
    """Whole-body tracking configuration specialized for Unitree G1."""

    motion_files: InitVar[tuple[str, ...] | None] = None
    commands: CommandsCfg = CommandsCfg(motion=WbtMotionCommandCfg())
    sim: SimCfg = SimCfg(dt=0.005, solver_iterations=3)
    scene: StandardSceneCfg = StandardSceneCfg(
        system_camera=SystemCameraCfg(distance=6.0, elevation=-20.0, azimuth=180.0),
        objs=StandardSceneObjsCfg(robot=UnitreeG129Dof()),
    )
    rewards: RewardsCfg = RewardsCfg(action_rate_l2=ActionRateRewardCfg(weight=-0.5))
    terminations: TerminationsCfg = TerminationsCfg(
        bad_body_z=BadBodyZTerminationCfg(
            threshold=0.25,
            body_names=(
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ),
        ),
    )

    def __post_init__(self, motion_files: tuple[str, ...] | None) -> None:
        super().__post_init__()
        if motion_files is not None:
            self.commands.motion.motion_files = motion_files
        self._set_tracked_body_names(G1_TRACKED_BODY_NAMES)
        self.commands.motion.reference_body_name = "torso_link"

        self.queries.data["undesired_contact_forces"] = BodyLinkNetContactForceQuery(
            body=self.scene.objs.robot.resolved_base_link_name,
            exclude_links=(
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ),
        )
