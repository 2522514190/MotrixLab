# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Robot-only carrying tracking with PC2 target clipping and physical perturbations."""

import os
from dataclasses import fields

import numpy as np

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.sim import LinkAngularVelocityQuery, LinkLinearVelocityQuery
from motrix_env_core.sim.model import (
    ActuatorKdQuery,
    BodyInertiasQuery,
    BodyMassesQuery,
    GeomFrictionQuery,
    LinkCenterOfMassQuery,
)
from motrix_env_core.sim.write import (
    ActuatorDampingWrite,
    ActuatorKpWrite,
    BodyAngularVelocityWrite,
    BodyLinearVelocityWrite,
    GeomFrictionWrite,
    LinkComWrite,
    LinkInertiaWrite,
    LinkMassWrite,
)
from motrix_envs.locomotion.wbt.cfg import RewardsCfg
from motrix_envs.locomotion.wbt.g1.common import MOTION_DIR, G1WbtEnvCfg
from motrix_envs.locomotion.wbt.mdp.rewards import JointPositionRewardCfg


@configclass
class RobotOnlyRandomizationCfg:
    """Dynamics draws are fixed per environment; pushes recur during rollout."""

    enabled: bool = True
    push_enabled: bool = False
    ground_sliding_friction: tuple[float, float] = (0.3, 1.2)
    push_interval_s: tuple[float, float] = (1.0, 3.0)
    push_linear: tuple[float, float, float] = (0.3, 0.3, 0.3)
    push_angular: tuple[float, float, float] = (0.78, 0.78, 0.78)
    link_mass_scale: tuple[float, float] = (1.0, 1.0)
    torso_com_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    kp_scale: tuple[float, float] = (1.0, 1.0)
    kd_scale: tuple[float, float] = (1.0, 1.0)


@configclass(kw_only=True)
class RobotOnlyCarryEnvCfg(G1WbtEnvCfg):
    randomization: RobotOnlyRandomizationCfg = RobotOnlyRandomizationCfg()
    play_training_perturbations: bool = True
    play_ground_friction: float | None = None
    evaluation: bool = False

    def for_play(self):
        cfg = super().for_play()
        cfg.evaluation = True
        cfg.randomization.enabled = cfg.play_training_perturbations
        if cfg.play_training_perturbations:
            # Restore training reset amplitudes after the base play configuration.
            for name in ("body_pos", "body_rot", "body_lin_vel", "body_rot_vel", "body_dof_pos"):
                getattr(cfg.sim_reset, name).noise_scale = getattr(self.sim_reset, name).noise_scale
        if not cfg.play_training_perturbations:
            for name in ("motion_ref_ori_b", "base_ang_vel", "dof_pos", "dof_vel"):
                getattr(cfg.observations.policy, name).noise.amplitude = 0.0
        if cfg.play_ground_friction is not None:
            if not np.isfinite(cfg.play_ground_friction) or cfg.play_ground_friction < 0:
                raise ValueError("play_ground_friction must be finite and nonnegative")
            cfg.scene.objs.floor.friction = (cfg.play_ground_friction, *cfg.scene.objs.floor.friction[1:])
        return cfg


@registry.envcfg("g1-29dof-wbt-largebox")
def make_g129dof_wbt_largebox_cfg() -> RobotOnlyCarryEnvCfg:
    """Track the official robot-only carrying motion with the original WBT reward recipe."""
    cfg = RobotOnlyCarryEnvCfg(motion_files=(str(MOTION_DIR / "sub3_largebox_003.npz"),))
    cfg.actions.joint_position.safety_clip_fraction = 0.95
    # As in the box ground-DR task, floor material wins contact mixing.
    cfg.scene.objs.floor.priority = 2
    cfg.queries.model["ground_action_friction"] = GeomFrictionQuery(name="floor")
    body = cfg.scene.objs.robot.resolved_base_link_name
    cfg.queries.data["root_lin_vel"] = LinkLinearVelocityQuery(link=body)
    cfg.queries.data["root_ang_vel"] = LinkAngularVelocityQuery(link=body)
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-opentrack406")
def make_g129dof_wbt_opentrack406_cfg() -> RobotOnlyCarryEnvCfg:
    """Use the official OpenTrack start prefix for the robot-only reference."""
    cfg = make_g129dof_wbt_largebox_cfg()
    cfg.commands.motion.motion_files = (str(MOTION_DIR / "sub3_largebox_003_opentrack_ik_start.npz"),)
    cfg.commands.motion.hold_at_clip_end = False
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-opentrack556-end3s")
def make_opentrack556_end3s_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_g129dof_wbt_opentrack406_cfg()
    cfg.commands.motion.motion_files = (str(MOTION_DIR / "sub3_largebox_003_opentrack_ik_start_end3s.npz"),)
    cfg.max_episode_seconds = 11.12
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-opentrack406-omni-reward-push")
def make_opentrack406_omni_reward_push_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_g129dof_wbt_opentrack406_cfg()
    cfg.rewards.motion_global_ref_position_error_exp.weight = 0.5
    cfg.rewards.motion_relative_body_position_error_exp.weight = 1.0
    cfg.rewards.action_rate_l2.weight = -0.1
    cfg.randomization.push_enabled = True
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-opentrack556-end3s-omni-reward")
def make_opentrack556_end3s_omni_reward_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_opentrack556_end3s_cfg()
    cfg.rewards = make_opentrack406_omni_reward_push_cfg().rewards
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-sbto649-hold3-omni-reward")
def make_sbto649_hold3_omni_reward_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_opentrack556_end3s_omni_reward_cfg()
    cfg.commands.motion.motion_files = (str(MOTION_DIR / "sub3_largebox_003_robot_only_sbto_hold3.npz"),)
    cfg.max_episode_seconds = 12.98
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward")
def make_opentrack674_loco_ik_end_sbto_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_sbto649_hold3_omni_reward_cfg()
    cfg.commands.motion.motion_files = (
        os.environ.get(
            "MOTRIX_OPENTRACK674_MOTION",
            str(MOTION_DIR / "sub3_largebox_003_robot_only_opentrack_loco_ik_end_sbto.npz"),
        ),
    )
    cfg.max_episode_seconds = 13.48
    return cfg


@configclass
class JointPositionCarryRewardsCfg(RewardsCfg):
    motion_joint_pos: JointPositionRewardCfg = JointPositionRewardCfg(sigma=0.4472, weight=2.0)


@registry.envcfg("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos")
def make_opentrack674_joint_pos_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_opentrack674_loco_ik_end_sbto_cfg()
    cfg.rewards = JointPositionCarryRewardsCfg(
        **{field.name: getattr(cfg.rewards, field.name) for field in fields(cfg.rewards)}
    )
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-delay01")
def make_opentrack674_joint_pos_delay01_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_opentrack674_joint_pos_cfg()
    cfg.actions.joint_position.action_delay_steps = (0, 1)
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-delay01-mass")
def make_opentrack674_delay01_mass_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_opentrack674_joint_pos_delay01_cfg()
    cfg.randomization.link_mass_scale = (0.8, 1.2)
    body = cfg.scene.objs.robot.resolved_base_link_name
    cfg.queries.model["robot_link_masses"] = BodyMassesQuery(body=body)
    cfg.queries.model["robot_link_inertias"] = BodyInertiasQuery(body=body)
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-delay01-mass-com")
def make_opentrack674_delay01_mass_com_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_opentrack674_delay01_mass_cfg()
    cfg.randomization.torso_com_offset = (0.02, 0.02, 0.02)
    cfg.queries.model["torso_com"] = LinkCenterOfMassQuery(name="torso_link")
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-delay01-mass-com-pd")
def make_opentrack674_delay01_mass_com_pd_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_opentrack674_delay01_mass_com_cfg()
    # Match the existing G1 velocity task's proportional/damping ranges.
    cfg.randomization.kp_scale = (0.9, 1.1)
    cfg.randomization.kd_scale = (0.9, 1.1)
    cfg.queries.model["actuator_kd"] = ActuatorKdQuery()
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-delay01-mass-com-pd-push")
def make_opentrack674_delay01_mass_com_pd_push_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_opentrack674_delay01_mass_com_pd_cfg()
    cfg.randomization.push_enabled = True
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-mass-no-delay")
def make_opentrack674_mass_no_delay_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_opentrack674_delay01_mass_cfg()
    cfg.actions.joint_position.action_delay_steps = (0, 0)
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-mass-com-no-delay")
def make_opentrack674_mass_com_no_delay_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_opentrack674_delay01_mass_com_cfg()
    cfg.actions.joint_position.action_delay_steps = (0, 0)
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-mass-com-pd-no-delay")
def make_opentrack674_mass_com_pd_no_delay_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_opentrack674_delay01_mass_com_pd_cfg()
    cfg.actions.joint_position.action_delay_steps = (0, 0)
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-mass-com-pd-push-no-delay")
def make_opentrack674_mass_com_pd_push_no_delay_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_opentrack674_delay01_mass_com_pd_push_cfg()
    cfg.actions.joint_position.action_delay_steps = (0, 0)
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-push")
def make_opentrack674_omni_reward_push_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_opentrack674_loco_ik_end_sbto_cfg()
    cfg.randomization.push_enabled = True
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-original-reward")
def make_opentrack674_original_reward_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_opentrack674_loco_ik_end_sbto_cfg()
    cfg.rewards = make_g129dof_wbt_opentrack406_cfg().rewards
    return cfg


def _use_sbto005_no_delay(cfg: RobotOnlyCarryEnvCfg) -> RobotOnlyCarryEnvCfg:
    default_motion = MOTION_DIR / "sub3_largebox_005_sbto_w_obj.npz"
    cfg.commands.motion.motion_files = (os.environ.get("MOTRIX_SBTO005_MOTION", str(default_motion)),)
    cfg.max_episode_seconds = 495 / 50
    cfg.actions.joint_position.action_delay_steps = (0, 0)
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-sbto005-joint-pos-no-delay")
def make_sbto005_no_delay_cfg() -> RobotOnlyCarryEnvCfg:
    return _use_sbto005_no_delay(make_opentrack674_joint_pos_delay01_cfg())


@registry.envcfg("g1-29dof-wbt-largebox-sbto005-joint-pos-no-delay-mass")
def make_sbto005_no_delay_mass_cfg() -> RobotOnlyCarryEnvCfg:
    return _use_sbto005_no_delay(make_opentrack674_delay01_mass_cfg())


@registry.envcfg("g1-29dof-wbt-largebox-sbto005-joint-pos-no-delay-mass-com")
def make_sbto005_no_delay_mass_com_cfg() -> RobotOnlyCarryEnvCfg:
    return _use_sbto005_no_delay(make_opentrack674_delay01_mass_com_cfg())


@registry.envcfg("g1-29dof-wbt-largebox-fixed-feet525-omni-reward-push")
def make_fixed_feet525_omni_reward_push_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_opentrack406_omni_reward_push_cfg()
    cfg.commands.motion.motion_files = (str(MOTION_DIR / "sub3_largebox_003_fixed_feet_start.npz"),)
    cfg.max_episode_seconds = 10.5
    return cfg


@registry.envcfg("g1-29dof-wbt-largebox-opentrack656-stand3s-end3s-omni-reward-push")
def make_opentrack656_stand3s_end3s_omni_reward_push_cfg() -> RobotOnlyCarryEnvCfg:
    cfg = make_opentrack406_omni_reward_push_cfg()
    cfg.commands.motion.motion_files = (str(MOTION_DIR / "sub3_largebox_003_opentrack_ik_start_stand3s_end3s.npz"),)
    cfg.max_episode_seconds = 13.12
    return cfg


@registry.env("g1-29dof-wbt-largebox-opentrack656-stand3s-end3s-omni-reward-push")
@registry.env("g1-29dof-wbt-largebox-fixed-feet525-omni-reward-push")
@registry.env("g1-29dof-wbt-largebox-opentrack406-omni-reward-push")
@registry.env("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-original-reward")
@registry.env("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-push")
@registry.env("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-mass-com-pd-push-no-delay")
@registry.env("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-delay01-mass-com-pd-push")
@registry.env("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-delay01-mass-com-pd")
@registry.env("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-delay01-mass-com")
@registry.env("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-delay01-mass")
@registry.env("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-delay01")
@registry.env("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos")
@registry.env("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward")
@registry.env("g1-29dof-wbt-largebox-sbto649-hold3-omni-reward")
@registry.env("g1-29dof-wbt-largebox-opentrack556-end3s-omni-reward")
@registry.env("g1-29dof-wbt-largebox-opentrack556-end3s")
@registry.env("g1-29dof-wbt-largebox-opentrack406")
@registry.env("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-mass-no-delay")
@registry.env("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-mass-com-no-delay")
@registry.env("g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-mass-com-pd-no-delay")
@registry.env("g1-29dof-wbt-largebox-sbto005-joint-pos-no-delay")
@registry.env("g1-29dof-wbt-largebox-sbto005-joint-pos-no-delay-mass")
@registry.env("g1-29dof-wbt-largebox-sbto005-joint-pos-no-delay-mass-com")
@registry.env("g1-29dof-wbt-largebox")
class RobotOnlyCarryEnv(ManagerEnv):
    """Keep per-lane ground draws and apply the box task's velocity-kick pushes."""

    def __init__(self, cfg, num_envs=1, backend=None, seed=None):
        super().__init__(cfg, num_envs=num_envs, backend=backend, seed=seed)
        self._dr_rng = np.random.default_rng(seed)
        self._ground_writes = self.sim.compile_writes({"friction": GeomFrictionWrite(("floor",))}, reset=False)
        friction = self._ground_writes.buffer("friction")
        friction[:, 0] = self.model.others["ground_action_friction"]
        dr = cfg.randomization
        if cfg.evaluation and cfg.play_ground_friction is not None:
            friction[:, 0, 0] = cfg.play_ground_friction
        elif dr.enabled:
            friction[:, 0, 0] = self._dr_rng.uniform(*dr.ground_sliding_friction, size=num_envs)
        body = cfg.scene.objs.robot.resolved_base_link_name
        self._push_writes = self.sim.compile_writes(
            {"linear": BodyLinearVelocityWrite((body,)), "angular": BodyAngularVelocityWrite((body,))},
            reset=False,
        )
        self._push_remaining = self._dr_rng.uniform(*dr.push_interval_s, size=num_envs)
        self._push_linear = np.asarray(dr.push_linear, dtype=np.float32)
        self._push_angular = np.asarray(dr.push_angular, dtype=np.float32)
        self._domain_writes = self._make_domain_writes(seed)

    def _make_domain_writes(self, seed):
        """Sample physical properties once and reapply after simulator resets.

        Mass and the full COM inertia tensor share one factor per positive-mass
        link. PD draws are independent per actuator; action scales retain their
        nominal calibration. No randomization is added to older task variants.
        """
        dr = self.cfg.randomization
        if not dr.enabled:
            return None
        writes = {}
        values = {}
        # Separate seeded streams preserve earlier stages' physical draws when
        # COM or PD is added, and leave floor/push sampling unchanged.
        mass_rng, com_rng, pd_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
        if dr.link_mass_scale != (1.0, 1.0):
            masses = self.model.others["robot_link_masses"]
            selected = np.flatnonzero(masses > 0.0)
            links = tuple(self.model.bodies["robot"].link_names[i] for i in selected)
            factors = mass_rng.uniform(*dr.link_mass_scale, size=(self.num_envs, len(links)))
            writes.update(mass=LinkMassWrite(links), inertia=LinkInertiaWrite(links))
            values["mass"] = masses[selected] * factors
            values["inertia"] = self.model.others["robot_link_inertias"][selected] * factors[..., None]
        if any(dr.torso_com_offset):
            writes["com"] = LinkComWrite(("torso_link",))
            amplitude = np.asarray(dr.torso_com_offset)
            values["com"] = self.model.others["torso_com"] + com_rng.uniform(
                -amplitude, amplitude, size=(self.num_envs, 1, 3)
            )
        actuators = tuple(spec.name for spec in self.model.actuators)
        for key, scale, write_type, nominal_key in (
            ("kp", dr.kp_scale, ActuatorKpWrite, "actuator_kp"),
            ("kd", dr.kd_scale, ActuatorDampingWrite, "actuator_kd"),
        ):
            if scale != (1.0, 1.0):
                writes[key] = write_type(actuators)
                values[key] = self.model.others[nominal_key] * pd_rng.uniform(
                    *scale, size=(self.num_envs, len(actuators))
                )
        if not writes:
            return None
        program = self.sim.compile_writes(writes, reset=False)
        for key, value in values.items():
            program.buffer(key)[:] = value
        return program

    def reset(self, env_ids, sim_reset_ids=None):
        super().reset(env_ids, sim_reset_ids)
        selected = env_ids if sim_reset_ids is None else np.union1d(env_ids, sim_reset_ids)
        if selected.size:
            self._ground_writes.execute(selected)
            if self._domain_writes is not None:
                self._domain_writes.execute(selected)
        if env_ids.size:
            self._push_remaining[env_ids] = self._dr_rng.uniform(
                *self.cfg.randomization.push_interval_s, size=env_ids.size
            )

    def physics_step(self):
        if self.cfg.randomization.enabled and self.cfg.randomization.push_enabled:
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
        super().physics_step()
