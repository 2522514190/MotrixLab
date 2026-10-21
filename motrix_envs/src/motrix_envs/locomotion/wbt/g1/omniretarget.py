# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0

"""OmniRetarget object-carrying recipe, independent of FastSAC task tuning.

Precedence: OmniRetarget IV > cited BeyondMimic S1/S4 > Holosoma WBT PPO.
See ``omniretarget_sources.md`` alongside this module for unspecified details.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np
from motrix_robots.unitree import UnitreeG129Dof

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import (
    BodyCfg,
    ContactReportField,
    ContactSensorCfg,
    ContactSensorReduce,
    PositionActuatorCfg,
    UrdfFileCfg,
)
from motrix_env_core.config.scene.base import SceneSensorsCfg
from motrix_env_core.manager import (
    ManagerActionsCfg,
    ManagerEnv,
    ManagerObservationGroupCfg,
    ManagerObservationsCfg,
    ManagerRewardsCfg,
    ManagerTerminationsCfg,
)
from motrix_env_core.mdp.observations import (
    ActionsObsCfg,
    BodyAngularVelocityObsCfg,
    BodyLinearVelocityObsCfg,
    UniformNoiseCfg,
)
from motrix_env_core.mdp.rewards import ActionRateRewardCfg
from motrix_env_core.sim import (
    LinkAngularVelocityQuery,
    LinkLinearVelocityQuery,
    LinkPositionQuery,
    LinkQuaternionQuery,
)
from motrix_env_core.sim.model import LinkCenterOfMassQuery
from motrix_env_core.sim.write import (
    BodyAngularVelocityWrite,
    BodyLinearVelocityWrite,
    LinkComWrite,
    LinkInertiaWrite,
    LinkMassWrite,
)
from motrix_envs.config.scene import StandardSceneCfg
from motrix_envs.locomotion.wbt.cfg import CommandsCfg, ResetCfg, WbtEnvCfg
from motrix_envs.locomotion.wbt.g1.common import G1_TRACKED_BODY_NAMES, LARGE_BOX_DIR, MOTION_DIR, G1WbtSceneObjsCfg
from motrix_envs.locomotion.wbt.mdp.command import WbtMotionCommandCfg
from motrix_envs.locomotion.wbt.mdp.object import (
    OBJECT_CHANNELS,
    ObjectLinearVelocityObsCfg,
    ObjectOrientationRewardCfg,
    ObjectPositionObsCfg,
    ObjectPositionRewardCfg,
    ObjectResetCfg,
)
from motrix_envs.locomotion.wbt.mdp.observations import (
    DofVelObsCfg,
    MotionJointObsCfg,
    RobotBodyPositionInReferenceFrameObsCfg,
)
from motrix_envs.locomotion.wbt.mdp.omniretarget import (
    BodyRot6dObsCfg,
    CalibratedActionCfg,
    CalibratedPositionObsCfg,
    ObjectRot6dObsCfg,
    PaperBodyTerminationCfg,
    ReferencePositionObsCfg,
    ReferenceRot6dObsCfg,
    SelfContactRewardCfg,
    StagedObjectTerminationCfg,
)
from motrix_envs.locomotion.wbt.mdp.reset import (
    BodyDofPosResetCfg,
    BodyLinVelResetCfg,
    BodyPosResetCfg,
    BodyRotResetCfg,
    BodyRotVelResetCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    DofLimitRewardCfg,
    GlobalBodyAngularVelocityRewardCfg,
    GlobalBodyLinearVelocityRewardCfg,
    GlobalRefOrientationRewardCfg,
    GlobalRefPositionRewardCfg,
    RelativeBodyOrientationRewardCfg,
    RelativeBodyPositionRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.tracking_metrics import LargeBoxTrackingMetricsCfg


@configclass
class OmniRewardsCfg(ManagerRewardsCfg):
    # BeyondMimic Table S1; object terms use body position/orientation settings.
    motion_global_ref_position_error_exp: GlobalRefPositionRewardCfg = GlobalRefPositionRewardCfg(weight=0.5, sigma=0.3)
    motion_global_ref_orientation_error_exp: GlobalRefOrientationRewardCfg = GlobalRefOrientationRewardCfg(
        weight=0.5, sigma=0.4
    )
    motion_relative_body_position_error_exp: RelativeBodyPositionRewardCfg = RelativeBodyPositionRewardCfg(
        weight=1.0, sigma=0.3
    )
    motion_relative_body_orientation_error_exp: RelativeBodyOrientationRewardCfg = RelativeBodyOrientationRewardCfg(
        weight=1.0, sigma=0.4
    )
    motion_global_body_lin_vel: GlobalBodyLinearVelocityRewardCfg = GlobalBodyLinearVelocityRewardCfg(
        weight=1.0, sigma=1.0
    )
    motion_global_body_ang_vel: GlobalBodyAngularVelocityRewardCfg = GlobalBodyAngularVelocityRewardCfg(
        weight=1.0, sigma=3.14
    )
    action_rate_l2: ActionRateRewardCfg = ActionRateRewardCfg(weight=-0.1)
    limits_dof_pos: DofLimitRewardCfg = DofLimitRewardCfg(weight=-10.0, soft_limit=0.9, cap=math.inf)
    undesired_contacts: SelfContactRewardCfg = SelfContactRewardCfg(weight=-0.1, threshold=1.0)
    object_global_ref_position_error_exp: ObjectPositionRewardCfg = ObjectPositionRewardCfg(weight=1.0, sigma=0.3)
    object_global_ref_orientation_error_exp: ObjectOrientationRewardCfg = ObjectOrientationRewardCfg(
        weight=1.0, sigma=0.4
    )


@configclass
class OmniPolicyCfg(ManagerObservationGroupCfg):
    motion_joint: MotionJointObsCfg = MotionJointObsCfg()
    motion_ref_pos_b: ReferencePositionObsCfg = ReferencePositionObsCfg()
    motion_ref_ori_b: ReferenceRot6dObsCfg = ReferenceRot6dObsCfg(noise=0.05)
    base_lin_vel: BodyLinearVelocityObsCfg = BodyLinearVelocityObsCfg(noise=UniformNoiseCfg(amplitude=0.5))
    base_ang_vel: BodyAngularVelocityObsCfg = BodyAngularVelocityObsCfg(noise=UniformNoiseCfg(amplitude=0.2))
    dof_pos: CalibratedPositionObsCfg = CalibratedPositionObsCfg(noise=0.01)
    dof_vel: DofVelObsCfg = DofVelObsCfg(noise=UniformNoiseCfg(amplitude=0.5))
    actions: ActionsObsCfg = ActionsObsCfg()


@configclass
class OmniValueCfg(OmniPolicyCfg):
    # Asymmetric critic: BM S1 per-body poses; Holosoma PPO object states.
    motion_ref_ori_b: ReferenceRot6dObsCfg = ReferenceRot6dObsCfg()
    base_lin_vel: BodyLinearVelocityObsCfg = BodyLinearVelocityObsCfg()
    base_ang_vel: BodyAngularVelocityObsCfg = BodyAngularVelocityObsCfg()
    dof_pos: CalibratedPositionObsCfg = CalibratedPositionObsCfg()
    dof_vel: DofVelObsCfg = DofVelObsCfg()
    robot_body_pos_b: RobotBodyPositionInReferenceFrameObsCfg = RobotBodyPositionInReferenceFrameObsCfg()
    robot_body_ori_b: BodyRot6dObsCfg = BodyRot6dObsCfg()
    obj_pos_b: ObjectPositionObsCfg = ObjectPositionObsCfg()
    obj_ori_b: ObjectRot6dObsCfg = ObjectRot6dObsCfg()
    obj_lin_vel_b: ObjectLinearVelocityObsCfg = ObjectLinearVelocityObsCfg()


@configclass
class OmniObservationsCfg(ManagerObservationsCfg):
    policy: OmniPolicyCfg = OmniPolicyCfg()
    value: OmniValueCfg = OmniValueCfg()


@configclass
class OmniActionsCfg(ManagerActionsCfg):
    joint_position: CalibratedActionCfg = CalibratedActionCfg(
        action_scale=0.25, action_scales_by_effort_limit_over_p_gain=True
    )


@configclass
class OmniResetCfg(ResetCfg):
    # Holosoma WBT command.py: initial-state perturbations.
    body_pos: BodyPosResetCfg = BodyPosResetCfg(noise=(0.05, 0.05, 0.01))
    body_rot: BodyRotResetCfg = BodyRotResetCfg(noise=(0.1, 0.1, 0.2))
    body_lin_vel: BodyLinVelResetCfg = BodyLinVelResetCfg(noise=(0.5, 0.5, 0.2))
    body_rot_vel: BodyRotVelResetCfg = BodyRotVelResetCfg(noise=(0.52, 0.52, 0.78))
    body_dof_pos: BodyDofPosResetCfg = BodyDofPosResetCfg(noise=0.1)
    object_state: ObjectResetCfg = ObjectResetCfg(body="baseLink", noise=0.05)


@configclass
class OmniTerminationsCfg(ManagerTerminationsCfg):
    body_tracking: PaperBodyTerminationCfg = PaperBodyTerminationCfg()
    object_tracking: StagedObjectTerminationCfg = StagedObjectTerminationCfg()
    tracking_metrics: LargeBoxTrackingMetricsCfg = LargeBoxTrackingMetricsCfg()


@configclass
class OmniRandomizationCfg:
    enabled: bool = True
    object_mass: tuple[float, float] = (0.1, 2.0)
    object_com: tuple[float, float, float] = (0.08, 0.08, 0.08)
    object_inertia_scale: tuple[float, float] = (0.5, 1.5)
    torso_com: tuple[float, float, float] = (0.025, 0.05, 0.075)
    joint_default_bias: float = 0.01
    push_interval_s: tuple[float, float] = (1.0, 3.0)
    # OmniRetarget gives scalar magnitudes; apply on all three axes.
    push_linear: tuple[float, float, float] = (0.3, 0.3, 0.3)
    push_angular: tuple[float, float, float] = (0.78, 0.78, 0.78)


@configclass
class SelfContactSensorsCfg(SceneSensorsCfg):
    links: tuple[str, ...] = ()
    geom_names: tuple[str, ...] = ()
    geom_links: tuple[int, ...] = ()

    def contact_pairs(self):
        for first in range(len(self.geom_names)):
            for second in range(first + 1, len(self.geom_names)):
                if self.geom_links[first] != self.geom_links[second]:
                    yield first, second

    def items(self):
        for first, second in self.contact_pairs():
            yield (
                f"self_contact_{first}_{second}",
                ContactSensorCfg(
                    geom1=self.geom_names[first],
                    geom2=self.geom_names[second],
                    data=[ContactReportField.force, ContactReportField.normal, ContactReportField.tangent],
                    reduce=ContactSensorReduce.netforce,
                ),
            )


@configclass
class ObjectCurriculumCfg:
    # The paper does not specify the quality gate and Holosoma WBT has no
    # object gate. Use its episode-length EMA, with an explicit local choice
    # of 80% horizon before latching object termination on.
    episode_window: int = 1000
    enable_at_horizon_fraction: float = 0.8


@configclass
class OmniRetargetEnvCfg(WbtEnvCfg):
    max_episode_seconds: float = 10.0
    sim: SimCfg = SimCfg(dt=0.005)
    ctrl_dt: float = 0.02
    commands: CommandsCfg = CommandsCfg(
        motion=WbtMotionCommandCfg(
            motion_files=(str(MOTION_DIR / "sub3_largebox_003_w_obj.npz"),),
            extension_channels=OBJECT_CHANNELS,
            reference_body_name="pelvis",
            adaptive_sampling_enabled=True,
            start_at_timestep_zero_prob=0.0,
            hold_at_clip_end=False,
            sequential_clips=False,
            uniform_ratio=0.1,
            alpha=0.001,
            kernel_size=3,
            kernel_lambda=0.8,
        )
    )
    scene: StandardSceneCfg = StandardSceneCfg(objs=G1WbtSceneObjsCfg(robot=UnitreeG129Dof()))
    actions: OmniActionsCfg = OmniActionsCfg()
    observations: OmniObservationsCfg = OmniObservationsCfg()
    rewards: OmniRewardsCfg = OmniRewardsCfg()
    terminations: OmniTerminationsCfg = OmniTerminationsCfg()
    sim_reset: OmniResetCfg = OmniResetCfg()
    randomization: OmniRandomizationCfg = OmniRandomizationCfg()
    object_curriculum: ObjectCurriculumCfg = ObjectCurriculumCfg()
    evaluation: bool = False
    randomize_initial_episode_progress: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self._set_tracked_body_names(G1_TRACKED_BODY_NAMES)
        body = self.sim_reset.object_state.body
        self.queries.data.update(
            object_pos=LinkPositionQuery(link=body),
            object_quat=LinkQuaternionQuery(link=body),
            object_lin_vel=LinkLinearVelocityQuery(link=body),
            object_ang_vel=LinkAngularVelocityQuery(link=body),
            root_lin_vel=LinkLinearVelocityQuery(link="pelvis"),
            root_ang_vel=LinkAngularVelocityQuery(link="pelvis"),
        )
        self.queries.model["torso_com"] = LinkCenterOfMassQuery(name="torso_link")

    def for_play(self):
        cfg = super().for_play()
        cfg.randomization.enabled = False
        cfg.evaluation = True
        cfg.randomize_initial_episode_progress = False
        cfg.sim_reset.object_state.noise_scale = 0.0
        # Actor noise is a training perturbation, not an evaluation perturbation.
        cfg.observations.policy.motion_ref_ori_b.noise = 0.0
        cfg.observations.policy.base_lin_vel.noise.amplitude = 0.0
        cfg.observations.policy.base_ang_vel.noise.amplitude = 0.0
        cfg.observations.policy.dof_pos.noise = 0.0
        cfg.observations.policy.dof_vel.noise.amplitude = 0.0
        return cfg


@registry.envcfg("g1-29dof-wbt-largebox-omniretarget")
def make_omniretarget_cfg() -> OmniRetargetEnvCfg:
    """OmniRetarget physical box carrying with the cited BeyondMimic PPO recipe.

    zh_CN: 采用 OmniRetarget / BeyondMimic 训练方案的 G1 物理搬箱任务。
    """
    cfg = OmniRetargetEnvCfg()
    robot_root = ET.parse(cfg.scene.objs.robot.model.file).getroot()
    bodies = robot_root.findall(".//body")
    collision_geoms = [
        (geom.attrib["name"], index)
        for index, body in enumerate(bodies)
        for geom in body.findall("geom")
        if geom.get("class") in ("collision", "foot")
    ]
    cfg.scene.sensors = SelfContactSensorsCfg(
        links=tuple(body.attrib["name"] for body in bodies),
        geom_names=tuple(name for name, _ in collision_geoms),
        geom_links=tuple(index for _, index in collision_geoms),
    )
    cfg.scene.objs.box = BodyCfg(
        model=UrdfFileCfg(
            file=LARGE_BOX_DIR / "large_box_omniretarget.urdf",
        ),
        base_link_name="baseLink",
    )
    # BM S1: kp = armature * (2*pi*10)^2, kd = 2*armature*2*(2*pi*10).
    # Armatures and torque limits follow Holosoma's G1 PPO robot preset.
    actuators = {}
    for name in cfg.scene.objs.robot.key_pose.joint_names:
        if "ankle" in name or "waist_roll" in name or "waist_pitch" in name:
            inertia, torque = 0.00721945, 50.0
        elif "hip_roll" in name or "knee" in name:
            inertia, torque = 0.025101925, 139.0
        elif "hip" in name or "waist_yaw" in name:
            inertia, torque = 0.010177520, 88.0
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            inertia, torque = 0.00425, 5.0
        else:
            inertia, torque = 0.003609725, 25.0
        omega = 2.0 * math.pi * 10.0
        actuators[name] = PositionActuatorCfg(
            joint_name=name,
            kp=inertia * omega**2,
            kv=4.0 * inertia * omega,
            ctrl_range=(-math.inf, math.inf),
            force_range=(-torque, torque),
        )
    cfg.scene.objs.robot.actuators = actuators
    return cfg


class OmniRetargetEnv(ManagerEnv):
    """Task-only randomization and curriculum hooks; ManagerEnv owns lifecycle."""

    def __init__(self, cfg, num_envs=1, backend=None, seed=None):
        super().__init__(cfg, num_envs=num_envs, backend=backend, seed=seed)
        self._dr_rng = np.random.default_rng(seed)
        self._average_episode_steps = 0.0
        self._elapsed_since_reset = np.zeros(num_envs, dtype=np.int64)
        self._gate = self.termination_manager.terms["object_tracking"].args[0]
        self._gate.enabled[0] = cfg.evaluation
        self._domain_writes = self.sim.compile_writes(
            {
                "mass": LinkMassWrite(("baseLink",)),
                "inertia": LinkInertiaWrite(("baseLink",)),
                "com": LinkComWrite(("baseLink", "torso_link")),
            },
            reset=False,
        )
        self._push_writes = self.sim.compile_writes(
            {
                "linear": BodyLinearVelocityWrite(("pelvis",)),
                "angular": BodyAngularVelocityWrite(("pelvis",)),
            },
            reset=False,
        )
        dr = cfg.randomization
        rng = self._dr_rng
        mass = rng.uniform(*dr.object_mass, size=num_envs) if dr.enabled else np.full(num_envs, 0.1)
        factor = rng.uniform(*dr.object_inertia_scale, size=num_envs) if dr.enabled else np.ones(num_envs)
        self._domain_writes.buffer("mass")[:, 0] = mass
        # Holosoma scales the nominal inertia with mass, then randomizes inertia.
        self._domain_writes.buffer("inertia")[:, 0, :3] = (mass * 0.02 * factor)[:, None]
        com = self._domain_writes.buffer("com")
        com[:, 0] = rng.uniform(-np.asarray(dr.object_com), dr.object_com, size=(num_envs, 3)) if dr.enabled else 0.0
        com[:, 1] = self.model.others["torso_com"]
        self._push_remaining = rng.uniform(*dr.push_interval_s, size=num_envs)
        self._push_linear = np.asarray(dr.push_linear, dtype=np.float32)
        self._push_angular = np.asarray(dr.push_angular, dtype=np.float32)

    def init_state(self):
        state = super().init_state()
        if self.cfg.randomize_initial_episode_progress:
            # Apply Holosoma's randomized initial timeout phases through the
            # existing environment state; the PPO provider remains unchanged.
            state.episode_steps[:] = self._dr_rng.integers(self.cfg.max_episode_steps, size=self.num_envs)
        return state

    def reset(self, env_ids, sim_reset_ids=None):
        # Update the Holosoma-style episode-length EMA only for actual episode
        # endings. Clip rematerialization and explicit resets cannot unlock it.
        if env_ids.size and self._state is not None and not self.cfg.evaluation:
            finished = env_ids[self._state.done[env_ids]]
            if finished.size:
                weight = min(finished.size / self.cfg.object_curriculum.episode_window, 1.0)
                mean = float(np.mean(self._elapsed_since_reset[finished]))
                self._average_episode_steps = (1.0 - weight) * self._average_episode_steps + weight * mean
                threshold = self.cfg.object_curriculum.enable_at_horizon_fraction * self.cfg.max_episode_steps
                self._gate.enabled[0] |= self._average_episode_steps >= threshold
        super().reset(env_ids, sim_reset_ids)
        if env_ids.size and self.cfg.randomization.enabled:
            # BM S1 specifies fresh robot calibration / COM draws per episode.
            # Clip rematerialization keeps the episode's robot parameters.
            dr = self.cfg.randomization
            self._domain_writes.buffer("com")[env_ids, 1] = self.model.others["torso_com"] + self._dr_rng.uniform(
                -np.asarray(dr.torso_com), dr.torso_com, size=(env_ids.size, 3)
            )
            self.action_terms["joint_position"].state.bias[env_ids] = self._dr_rng.uniform(
                -dr.joint_default_bias, dr.joint_default_bias, size=(env_ids.size, self.num_actuators)
            )
        selected = env_ids if sim_reset_ids is None else np.concatenate((env_ids, sim_reset_ids))
        if selected.size:
            # Native reset restores overrides; reapply physical parameters to
            # reset lanes, including motion-clip boundaries.
            self._domain_writes.execute(selected)
        if env_ids.size:
            self._elapsed_since_reset[env_ids] = 0
            self._push_remaining[env_ids] = self._dr_rng.uniform(
                *self.cfg.randomization.push_interval_s, size=env_ids.size
            )

    def physics_step(self):
        if self.cfg.randomization.enabled:
            self._push_remaining -= self.cfg.ctrl_dt
            selected = np.flatnonzero(self._push_remaining <= 0.0)
            if selected.size:
                self._refresh_sim_reads(selected)
                for key, source, amplitude in (
                    ("linear", "root_lin_vel", self._push_linear),
                    ("angular", "root_ang_vel", self._push_angular),
                ):
                    self._push_writes.buffer(key)[selected, 0] = self.sim_data[source][selected] + self._dr_rng.uniform(
                        -amplitude, amplitude, size=(selected.size, 3)
                    )
                self._push_writes.execute(selected)
                self._push_remaining[selected] = self._dr_rng.uniform(
                    *self.cfg.randomization.push_interval_s, size=selected.size
                )
        self._elapsed_since_reset += 1
        super().physics_step()


registry.env("g1-29dof-wbt-largebox-omniretarget")(OmniRetargetEnv)
