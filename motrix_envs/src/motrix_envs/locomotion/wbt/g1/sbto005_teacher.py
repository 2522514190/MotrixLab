# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Privileged FastSAC teacher for the physical SBTO 005 box."""

import os
from copy import deepcopy
from math import isfinite
from pathlib import Path

import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import (
    BodyCfg,
    ContactReportField,
    ContactSensorCfg,
    ContactSensorReduce,
    SceneSensorsCfg,
    UrdfFileCfg,
    UrdfGeomCfg,
)
from motrix_env_core.manager.env import ManagerEnv
from motrix_env_core.manager.terminations import ManagerTerminationsCfg
from motrix_env_core.mdp.rewards import ActionRateRewardCfg
from motrix_env_core.sim import (
    BodyLinkNetContactForceQuery,
    LinkAngularVelocityQuery,
    LinkLinearVelocityQuery,
    LinkPositionQuery,
    LinkQuaternionQuery,
    SensorValuesQuery,
)
from motrix_env_core.sim.model import (
    ActuatorKdQuery,
    BodyMassesQuery,
    GeomFrictionQuery,
    LinkCenterOfMassQuery,
    LinkInertiaQuery,
)
from motrix_env_core.sim.write import GeomFrictionWrite, LinkComWrite, LinkInertiaWrite, LinkMassWrite
from motrix_env_core.torch_manager.runtime import TorchTaskProgram
from motrix_envs.config.scene.standard import StandardSceneObjsCfg
from motrix_envs.locomotion.wbt.cfg import CommandsCfg, ObservationsCfg, ResetCfg, RewardsCfg
from motrix_envs.locomotion.wbt.g1.common import G1WbtEnvCfg
from motrix_envs.locomotion.wbt.mdp.command import WbtMotionCommandCfg
from motrix_envs.locomotion.wbt.mdp.rewards import (
    GlobalBodyAngularVelocityRewardCfg,
    GlobalBodyLinearVelocityRewardCfg,
    GlobalRefOrientationRewardCfg,
    GlobalRefPositionRewardCfg,
    RelativeBodyOrientationRewardCfg,
    RelativeBodyPositionRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.sbto005_teacher import (
    CONTACT_CHANNEL,
    CONTACT_NAMES,
    HAND_LINKS,
    OBJECT_CHANNELS,
    BoxPoseObsCfg,
    BoxResetCfg,
    BoxRewardCfg,
    BoxVelocityErrorObsCfg,
    HandContactRewardCfg,
    JointAccelerationSquaredCfg,
    JointPositionRewardCfg,
    JointTorqueSquaredCfg,
    ReferenceBodyOrientationObsCfg,
)
from motrix_envs.locomotion.wbt.mdp.terminations import (
    BadBodyZTerminationCfg,
    BadRefOrientationTerminationCfg,
    BadRefZTerminationCfg,
)

TASK = "g1-29dof-wbt-largebox-sbto005-teacher"
MOTION_ENV_VAR = "MOTRIX_SBTO005_MOTION"
PLAY_PERTURBATIONS_ENV_VAR = "MOTRIX_SBTO005_PLAY_PERTURBATIONS"
HAND_CONTACT_CURRICULUM_ENV_VAR = "MOTRIX_SBTO005_HAND_CONTACT_CURRICULUM"
HAND_CONTACT_WEIGHT_ENV_VAR = "MOTRIX_SBTO005_HAND_CONTACT_WEIGHT"
HAND_CONTACT_REWARD_ENV_VAR = "MOTRIX_SBTO005_HAND_CONTACT_REWARD"
BOX_XY_DR_ENV_VAR = "MOTRIX_SBTO005_BOX_XY_DR"
CLIP_FRAMES = 495
CLIP_FPS = 50
BOX_URDF = Path(__file__).resolve().parent.parent / "assets" / "objects" / "teacher005" / "box.urdf"
BOX_LINK = "baseLink"
BOX_GEOM = "largebox_collision"
BOX_HALF_EXTENTS = (0.1025, 0.0625, 0.14)
BOX_MASS = 0.6
BOX_FRICTION = (0.6, 0.0001, 0.0001)
HAND_GEOMS = ("left_hand_collision", "right_hand_collision")
END_EFFECTORS = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")


@configclass
class TeacherRewardsCfg(RewardsCfg):
    motion_global_ref_position_error_exp: GlobalRefPositionRewardCfg = GlobalRefPositionRewardCfg(
        weight=0.5, sigma=0.4472
    )
    motion_global_ref_orientation_error_exp: GlobalRefOrientationRewardCfg = GlobalRefOrientationRewardCfg(
        weight=0.5, sigma=0.5774
    )
    motion_relative_body_position_error_exp: RelativeBodyPositionRewardCfg = RelativeBodyPositionRewardCfg(
        weight=1.0, sigma=0.4472
    )
    motion_relative_body_orientation_error_exp: RelativeBodyOrientationRewardCfg = RelativeBodyOrientationRewardCfg(
        weight=1.0, sigma=0.5774
    )
    motion_global_body_lin_vel: GlobalBodyLinearVelocityRewardCfg = GlobalBodyLinearVelocityRewardCfg(
        weight=1.0, sigma=1.4142
    )
    motion_global_body_ang_vel: GlobalBodyAngularVelocityRewardCfg = GlobalBodyAngularVelocityRewardCfg(
        weight=1.0, sigma=4.4721
    )
    motion_joint_pos: JointPositionRewardCfg = JointPositionRewardCfg(weight=2.0, sigma=0.4472)
    action_rate_l2: ActionRateRewardCfg = ActionRateRewardCfg(weight=-0.1)
    box_position: BoxRewardCfg = BoxRewardCfg(weight=1.0, sigma=0.4247, quantity="position")
    box_orientation: BoxRewardCfg = BoxRewardCfg(weight=1.0, sigma=0.5372, quantity="orientation")
    box_linear_velocity: BoxRewardCfg = BoxRewardCfg(weight=1.0, sigma=0.7071, quantity="linear_velocity")
    box_angular_velocity: BoxRewardCfg = BoxRewardCfg(weight=1.0, sigma=2.2361, quantity="angular_velocity")
    left_hand_contact: HandContactRewardCfg = HandContactRewardCfg(weight=5.0, hand=0)
    right_hand_contact: HandContactRewardCfg = HandContactRewardCfg(weight=5.0, hand=1)
    joint_acceleration: JointAccelerationSquaredCfg = JointAccelerationSquaredCfg(weight=-2.5e-7)
    joint_torque: JointTorqueSquaredCfg = JointTorqueSquaredCfg(weight=-1e-5)


@configclass
class TeacherPolicyCfg(ObservationsCfg.PolicyCfg):
    box_pose: BoxPoseObsCfg = BoxPoseObsCfg(noise=0.01)
    box_pose_error: BoxPoseObsCfg = BoxPoseObsCfg(error=True, noise=0.01)


@configclass
class TeacherValueCfg(ObservationsCfg.ValueCfg):
    robot_body_ori_b: ReferenceBodyOrientationObsCfg = ReferenceBodyOrientationObsCfg()
    box_pose: BoxPoseObsCfg = BoxPoseObsCfg()
    box_pose_error: BoxPoseObsCfg = BoxPoseObsCfg(error=True)
    box_velocity_error: BoxVelocityErrorObsCfg = BoxVelocityErrorObsCfg()


@configclass
class TeacherObservationsCfg(ObservationsCfg):
    policy: TeacherPolicyCfg = TeacherPolicyCfg()
    value: TeacherValueCfg = TeacherValueCfg()


@configclass
class TeacherTerminationsCfg(ManagerTerminationsCfg):
    bad_ref_z: BadRefZTerminationCfg = BadRefZTerminationCfg(threshold=0.25)
    bad_ref_ori: BadRefOrientationTerminationCfg = BadRefOrientationTerminationCfg(threshold=0.8)
    bad_body_z: BadBodyZTerminationCfg = BadBodyZTerminationCfg(threshold=0.25, body_names=END_EFFECTORS)


@configclass
class TeacherResetCfg(ResetCfg):
    object_state: BoxResetCfg = BoxResetCfg(body=BOX_LINK)


@configclass
class TeacherSceneObjsCfg(StandardSceneObjsCfg):
    box: BodyCfg = BodyCfg(
        model=UrdfFileCfg(
            file=BOX_URDF,
            geoms=[UrdfGeomCfg(geom_name=BOX_GEOM, friction=BOX_FRICTION, condim=3, priority=3)],
        ),
        base_link_name=BOX_LINK,
    )


@configclass
class TeacherSceneSensorsCfg(SceneSensorsCfg):
    left_hand_box: ContactSensorCfg = ContactSensorCfg(
        geom1=HAND_GEOMS[0],
        geom2=BOX_GEOM,
        data=[ContactReportField.force],
        reduce=ContactSensorReduce.netforce,
    )
    right_hand_box: ContactSensorCfg = ContactSensorCfg(
        geom1=HAND_GEOMS[1],
        geom2=BOX_GEOM,
        data=[ContactReportField.force],
        reduce=ContactSensorReduce.netforce,
    )


@configclass
class TeacherRandomizationCfg:
    enabled: bool = True
    ground_sliding_friction: tuple[float, float] = (0.3, 1.2)
    torso_com_offset: float = 0.02
    box_mass_scale: tuple[float, float] = (0.8, 1.25)
    box_sliding_friction: tuple[float, float] = (0.4, 0.8)


@configclass(frozen=True, kw_only=True)
class HandContactCurriculumCfg:
    start_iteration: int
    end_iteration: int
    final_weight: float

    def __post_init__(self) -> None:
        if not 0 <= self.start_iteration < self.end_iteration:
            raise ValueError("Expected 0 <= start_iteration < end_iteration.")
        if not isfinite(self.final_weight) or self.final_weight < 0.0:
            raise ValueError("Expected a finite, nonnegative final_weight.")


@configclass(kw_only=True)
class Sbto005TeacherCfg(G1WbtEnvCfg):
    max_episode_seconds: float = CLIP_FRAMES / CLIP_FPS
    commands: CommandsCfg = CommandsCfg(
        motion=WbtMotionCommandCfg(extension_channels=(*OBJECT_CHANNELS, CONTACT_CHANNEL), hold_at_clip_end=True)
    )
    sim_reset: TeacherResetCfg = TeacherResetCfg()
    rewards: TeacherRewardsCfg = TeacherRewardsCfg()
    terminations: TeacherTerminationsCfg = TeacherTerminationsCfg()
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    randomization: TeacherRandomizationCfg = TeacherRandomizationCfg()
    hand_contact_curriculum: HandContactCurriculumCfg | None = None

    def for_play(self) -> "Sbto005TeacherCfg":
        perturbations = os.environ.get(PLAY_PERTURBATIONS_ENV_VAR, "0")
        if perturbations not in ("0", "1"):
            raise ValueError(f"{PLAY_PERTURBATIONS_ENV_VAR} must be 0 or 1.")
        cfg = super().for_play()
        cfg.commands.motion.sequential_clips = False
        cfg.commands.motion.hold_at_clip_end = True
        cfg.hand_contact_curriculum = None
        if perturbations == "1":
            cfg.sim_reset = deepcopy(self.sim_reset)
            return cfg
        policy = cfg.observations.policy
        for term in (policy.motion_ref_ori_b, policy.base_ang_vel, policy.dof_pos, policy.dof_vel):
            term.noise.amplitude = 0.0
        policy.box_pose.noise = 0.0
        policy.box_pose_error.noise = 0.0
        cfg.sim_reset.object_state.initial_xy_noise = 0.0
        cfg.randomization.enabled = False
        return cfg


def _motion_path() -> Path:
    source = os.environ.get(MOTION_ENV_VAR)
    if not source:
        raise ValueError(f"Set {MOTION_ENV_VAR} to the verified SBTO 005 box-motion NPZ.")
    path = Path(source).expanduser().resolve()
    with np.load(path, allow_pickle=False) as motion:
        if int(motion["num_frames"]) != CLIP_FRAMES or int(motion["fps"]) != CLIP_FPS:
            raise ValueError(f"SBTO 005 requires {CLIP_FRAMES} frames at {CLIP_FPS} Hz: {path}")
        if tuple(motion["ext_contact_computed_names"]) != CONTACT_NAMES:
            raise ValueError(f"SBTO 005 contact labels must be ordered as {CONTACT_NAMES}: {path}")
        shapes = ((3,), (4,), (3,), (3,), (len(CONTACT_NAMES),))
        for channel, shape in zip((*OBJECT_CHANNELS, CONTACT_CHANNEL), shapes, strict=True):
            if motion[f"ext_{channel}"].shape != (CLIP_FRAMES, *shape):
                raise ValueError(f"Unexpected shape for SBTO 005 channel ext_{channel}: {path}")
        for field, expected in (
            ("ext_object_geom_size", BOX_HALF_EXTENTS),
            ("ext_object_mass", (BOX_MASS,)),
            ("ext_object_friction", BOX_FRICTION),
        ):
            if not np.allclose(motion[field], expected):
                raise ValueError(f"SBTO 005 box metadata {field} must match {expected}: {path}")
    return path


def _hand_contact_curriculum() -> HandContactCurriculumCfg | None:
    value = os.environ.get(HAND_CONTACT_CURRICULUM_ENV_VAR)
    if value is None:
        return None
    try:
        start, end, weight = value.split(",")
        return HandContactCurriculumCfg(start_iteration=int(start), end_iteration=int(end), final_weight=float(weight))
    except ValueError as exc:
        raise ValueError(
            f"{HAND_CONTACT_CURRICULUM_ENV_VAR} must specify start_iteration,end_iteration,final_weight. {exc}"
        ) from exc


def _box_initial_xy_noise() -> float:
    message = f"{BOX_XY_DR_ENV_VAR} must be a finite nonnegative half-range in meters."
    try:
        amplitude = float(os.environ.get(BOX_XY_DR_ENV_VAR, "0"))
    except ValueError as exc:
        raise ValueError(message) from exc
    if not isfinite(amplitude) or amplitude < 0.0:
        raise ValueError(message)
    return amplitude


@registry.envcfg(TASK)
def make_sbto005_teacher_cfg() -> Sbto005TeacherCfg:
    fixed_weight = os.environ.get(HAND_CONTACT_WEIGHT_ENV_VAR)
    contact_reward = os.environ.get(HAND_CONTACT_REWARD_ENV_VAR, "binary")
    if contact_reward not in ("binary", "grasp"):
        raise ValueError(f"Expected {HAND_CONTACT_REWARD_ENV_VAR} to be 'binary' or 'grasp'.")
    cfg = Sbto005TeacherCfg(
        motion_files=(str(_motion_path()),),
        hand_contact_curriculum=_hand_contact_curriculum() if fixed_weight is None else None,
    )
    if fixed_weight is not None:
        message = f"{HAND_CONTACT_WEIGHT_ENV_VAR} must be a finite nonnegative weight."
        try:
            weight = float(fixed_weight)
        except ValueError as exc:
            raise ValueError(message) from exc
        if not isfinite(weight) or weight < 0.0:
            raise ValueError(message)
        cfg.rewards.left_hand_contact.weight = weight
        cfg.rewards.right_hand_contact.weight = weight
    cfg.sim_reset.object_state.initial_xy_noise = _box_initial_xy_noise()
    if cfg.sim_reset.object_state.initial_xy_noise:
        # Mix full approaches into training while keeping reference initialization later in the clip.
        cfg.commands.motion.start_at_timestep_zero_prob = 0.5
    cfg.scene.objs = TeacherSceneObjsCfg(robot=cfg.scene.objs.robot)
    cfg.scene.sensors = TeacherSceneSensorsCfg()
    contact_query = "hand_box_contact_forces"
    if contact_reward == "grasp":
        contact_query = "hand_box_contact_points"
        for sensor, reward in (
            (cfg.scene.sensors.left_hand_box, cfg.rewards.left_hand_contact),
            (cfg.scene.sensors.right_hand_box, cfg.rewards.right_hand_contact),
        ):
            # MotrixSim reports the contact point on geom1. Score the box surface.
            sensor.geom1, sensor.geom2 = sensor.geom2, sensor.geom1
            sensor.num = 4
            sensor.data = [ContactReportField.found, ContactReportField.force, ContactReportField.pos]
            sensor.reduce = ContactSensorReduce.none
            reward.box_height = 2 * BOX_HALF_EXTENTS[2]
    cfg.scene.objs.floor.priority = 2
    cfg.actions.joint_position.safety_clip_fraction = 0.95
    cfg.queries.data.update(
        object_pos=LinkPositionQuery(link=BOX_LINK),
        object_quat=LinkQuaternionQuery(link=BOX_LINK),
        object_lin_vel=LinkLinearVelocityQuery(link=BOX_LINK),
        object_ang_vel=LinkAngularVelocityQuery(link=BOX_LINK),
        undesired_contact_forces=BodyLinkNetContactForceQuery(
            body="pelvis", exclude_links=(*END_EFFECTORS[:2], *HAND_LINKS)
        ),
    )
    cfg.queries.data[contact_query] = SensorValuesQuery(sensors=("left_hand_box", "right_hand_box"))
    cfg.queries.model.update(
        actuator_kd=ActuatorKdQuery(),
        floor_friction=GeomFrictionQuery(name="floor"),
        box_friction=GeomFrictionQuery(name=BOX_GEOM),
        box_mass=BodyMassesQuery(body=BOX_LINK),
        box_inertia=LinkInertiaQuery(name=BOX_LINK),
        torso_com=LinkCenterOfMassQuery(name="torso_link"),
    )
    return cfg


@registry.env(TASK)
class Sbto005TeacherEnv(ManagerEnv):
    """Run physical DR and cut rollouts at the last scored reference frame."""

    def __init__(self, cfg: Sbto005TeacherCfg, num_envs=1, backend=None, seed=None, *, headless=False, record=None):
        super().__init__(cfg, num_envs, backend, seed, headless=headless, record=record)
        self._domain_writes = self._sample_domain() if cfg.randomization.enabled else None
        self._initializing_episode_batch = False
        self._episode_start_frame = torch.zeros(num_envs, dtype=torch.long)
        self._clip_truncated = torch.zeros(num_envs, dtype=torch.bool)
        self._time_truncated = torch.zeros(num_envs, dtype=torch.bool)
        self._hand_contact_weights: tuple[torch.Tensor, ...] = ()
        self._previous_joint_velocity = self._reward_terms["joint_acceleration"].args[0]
        self._pd_effort_squared = self._reward_terms["joint_torque"].args[0]
        joint_names = self.model.bodies["robot"].joint_names
        self._actuator_joint_indices = [joint_names.index(spec.target_name) for spec in self.model.actuators]
        self._force_limits = torch.tensor([spec.force_range for spec in self.model.actuators], dtype=torch.float32).T

    def _sample_domain(self):
        dr = self.cfg.randomization
        generator = torch.Generator().manual_seed(self._build_seed)
        program = self.sim.compile_writes(
            {
                "floor_friction": GeomFrictionWrite(("floor",)),
                "box_friction": GeomFrictionWrite((BOX_GEOM,)),
                "box_mass": LinkMassWrite((BOX_LINK,)),
                "box_inertia": LinkInertiaWrite((BOX_LINK,)),
                "torso_com": LinkComWrite(("torso_link",)),
            },
            reset=False,
        )
        nominal = self.model.others
        for name, (low, high) in (
            ("floor_friction", dr.ground_sliding_friction),
            ("box_friction", dr.box_sliding_friction),
        ):
            friction = program.buffer(name)
            friction[:, 0] = nominal[name]
            friction[:, 0, 0].uniform_(low, high, generator=generator)
        mass_scale = torch.empty((self.num_envs, 1)).uniform_(*dr.box_mass_scale, generator=generator)
        program.buffer("box_mass")[:] = nominal["box_mass"] * mass_scale
        program.buffer("box_inertia")[:, 0] = nominal["box_inertia"] * mass_scale
        amp = dr.torso_com_offset
        com_offset = torch.empty((self.num_envs, 1, 3)).uniform_(-amp, amp, generator=generator)
        program.buffer("torso_com")[:] = nominal["torso_com"] + com_offset
        return program

    def init_state(self):
        self._initializing_episode_batch = True
        try:
            return super().init_state()
        finally:
            self._initializing_episode_batch = False

    def _build_task_program(self) -> TorchTaskProgram:
        program = super()._build_task_program()
        names = ("left_hand_contact", "right_hand_contact")
        indices = {term.name: term.index for term in self.manager_layout.rewards}
        self._hand_contact_weights = tuple(program.reward_weights[indices[name]] for name in names)
        return program

    def set_training_iteration(self, iteration: int) -> None:
        curriculum = self.cfg.hand_contact_curriculum
        if curriculum is None:
            return
        progress = (iteration - curriculum.start_iteration) / (curriculum.end_iteration - curriculum.start_iteration)
        progress = min(max(progress, 0.0), 1.0)
        initial_weights = (self.cfg.rewards.left_hand_contact.weight, self.cfg.rewards.right_hand_contact.weight)
        for target, initial in zip(self._hand_contact_weights, initial_weights, strict=True):
            target.fill_(initial + (curriculum.final_weight - initial) * progress)

    def _make_metrics_view(self):
        metrics = super()._make_metrics_view()
        metrics.update(
            episode_start_frame=self._episode_start_frame,
            episode_clip_end=self._clip_truncated,
            episode_timeout=self._time_truncated,
        )
        if self._hand_contact_weights:
            metrics["hand_contact_weight"] = self._hand_contact_weights[0]
        return metrics

    def _reset_sim_rows(self, env_ids: torch.Tensor, episode_reset: torch.Tensor) -> None:
        if self._initializing_episode_batch:
            # Initial reset uses synthetic done flags, which should not update the adaptive failure sampler.
            self._state.terminated[env_ids] = False
        super()._reset_sim_rows(env_ids, episode_reset)
        episode_ids = env_ids[episode_reset]
        self._episode_start_frame[episode_ids] = self.command_terms["motion"].steps[episode_ids, 0]
        if self._domain_writes is not None:
            self._domain_writes.execute(env_ids)

    def physics_step(self) -> None:
        self._previous_joint_velocity.copy_(self.sim_data["robot_dof_vel"])
        indices = self._actuator_joint_indices
        target = self._action_writes.buffer("joint_position")
        kp = self.model.others["actuator_kp"]
        kd = self.model.others["actuator_kd"]
        effort = (
            kp * (target - self.sim_data["robot_dof_pos"][:, indices]) - kd * self._previous_joint_velocity[:, indices]
        )
        effort.clamp_(*self._force_limits)
        self._pd_effort_squared.copy_(effort.square().sum(dim=-1))
        super().physics_step()

    def _update_truncate(self) -> None:
        super()._update_truncate()
        state = self._state
        motion = self.command_terms["motion"]
        clip_ended = motion.clip_ended[:, 0]
        self._clip_truncated.copy_(clip_ended & ~state.terminated)
        self._time_truncated.copy_(state.truncated & ~state.terminated & ~clip_ended)
        state.truncated.copy_(self._clip_truncated | self._time_truncated)
        # The Manager advances after scoring. Attribute failures to the scored reference frame.
        motion.steps[:, 0] -= (state.terminated & ~clip_ended).to(motion.steps.dtype)
