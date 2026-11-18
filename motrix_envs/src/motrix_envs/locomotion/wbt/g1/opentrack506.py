"""Physical OpenTrack carry with the robot-only actor and corrected asymmetric critic."""

from itertools import combinations
from xml.etree import ElementTree

import numpy as np

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import ContactReportField, ContactSensorCfg, ContactSensorReduce, SceneSensorsCfg
from motrix_env_core.manager import ManagerContext, RewardTerm, RewardTermCfg
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.sim import GeomFrictionQuery, SensorValuesQuery
from motrix_env_core.sim.write import GeomFrictionWrite
from motrix_envs.locomotion.wbt.g1.common import MOTION_DIR
from motrix_envs.locomotion.wbt.g1.largebox import make_opentrack406_omni_reward_push_cfg
from motrix_envs.locomotion.wbt.g1.largebox_object import LargeBoxObservationsCfg
from motrix_envs.locomotion.wbt.g1.omniretarget import OmniRetargetEnv, OmniRetargetEnvCfg, make_omniretarget_cfg

TASK = "g1-29dof-wbt-largebox-opentrack506-critic-v2"


@configclass
class SelfContactSensors(SceneSensorsCfg):
    pairs: dict[str, ContactSensorCfg] = {}

    def items(self):
        yield from self.pairs.items()


@dispatch
def self_contact_count(ctx: ManagerContext, owners: np.ndarray, threshold: np.float32) -> float:
    forces = ctx.sim["self_contact_forces"]
    touched = np.zeros(owners.size, dtype=np.bool_)
    for index in range(owners.shape[0]):
        x, y, z = forces[index * 3 : index * 3 + 3]
        if x*x + y*y + z*z > threshold * threshold:
            for side in range(2):
                body = owners[index, side]
                if body >= 0:
                    touched[body] = True
    return float(np.sum(touched))


@configclass(kw_only=True)
class SelfContactReward(RewardTermCfg):
    owners: tuple[tuple[int, int], ...]
    threshold: float = 1.0

    def __call__(self, ctx):
        return RewardTerm(self_contact_count, np.asarray(self.owners, dtype=np.int32), np.float32(self.threshold))


def configure_self_contacts(cfg):
    robot = cfg.scene.objs.robot
    links = ElementTree.parse(robot.model.file).findall(".//worldbody//body")
    excluded = set(cfg.queries.data.pop("undesired_contact_forces").exclude_links)
    geoms = [
        (robot.resolve_name(geom.attrib["name"]), index, body.attrib["name"])
        for index, body in enumerate(links)
        for geom in body.findall("geom[@class='collision']")
    ]
    sensors = SelfContactSensors()
    owners = []
    for (a, i, left), (b, j, right) in combinations(geoms, 2):
        if i == j or (left in excluded and right in excluded):
            continue
        sensors.pairs[f"self_{a}_{b}"] = ContactSensorCfg(
            geom1=a, geom2=b, data=[ContactReportField.force], reduce=ContactSensorReduce.netforce
        )
        owners.append((i if left not in excluded else -1, j if right not in excluded else -1))
    cfg.scene.sensors = sensors
    cfg.queries.data["self_contact_forces"] = SensorValuesQuery(sensors=tuple(sensors.pairs))
    cfg.rewards.undesired_contacts = SelfContactReward(weight=-0.1, owners=tuple(owners))


@registry.envcfg(TASK)
def make_opentrack506_cfg() -> OmniRetargetEnvCfg:
    cfg = make_omniretarget_cfg()
    baseline = make_opentrack406_omni_reward_push_cfg()
    cfg.actions = baseline.actions
    cfg.observations = LargeBoxObservationsCfg()
    cfg.observations.value.robot_body_ori_b.corrected_reference = True
    cfg.rewards.action_rate_l2.weight = -0.1
    cfg.randomization.torso_com = (0.0, 0.0, 0.0)
    cfg.randomize_initial_episode_progress = False
    cfg.commands.motion.motion_files = (str(MOTION_DIR / "sub3_largebox_005_sbto_w_obj.npz"),)
    cfg.commands.motion.kernel_size = baseline.commands.motion.kernel_size
    cfg.max_episode_seconds = 11.0
    cfg.scene.objs.floor.priority = 2
    cfg.queries.model["ground_action_friction"] = GeomFrictionQuery(name="floor")
    configure_self_contacts(cfg)
    return cfg


@registry.env(TASK)
class OpenTrackCarryEnv(OmniRetargetEnv):
    def __init__(self, cfg, num_envs=1, backend=None, seed=None):
        super().__init__(cfg, num_envs=num_envs, backend=backend, seed=seed)
        self._training_iteration = 0
        self._gate_iteration = -1
        self._ground = self.sim.compile_writes({"floor": GeomFrictionWrite(("floor",))}, reset=False)
        friction = self._ground.buffer("floor")
        friction[:, 0] = self.model.others["ground_action_friction"]
        if cfg.play_ground_friction is not None and cfg.evaluation:
            friction[:, 0, 0] = cfg.play_ground_friction
        elif cfg.randomization.enabled:
            friction[:, 0, 0] = self._dr_rng.uniform(0.6, 1.6, num_envs)

    def set_training_iteration(self, iteration):
        self._training_iteration = int(iteration)

    def compute_transition(self, state):
        state = super().compute_transition(state)
        state.metrics["carry_episode_steps_ema"] = self._average_episode_steps
        state.metrics["carry_gate_first_iteration"] = self._gate_iteration
        return state

    def reset(self, env_ids, sim_reset_ids=None):
        super().reset(env_ids, sim_reset_ids)
        selected = env_ids if sim_reset_ids is None else np.union1d(env_ids, sim_reset_ids)
        if selected.size:
            self._ground.execute(selected)
        if not self.cfg.evaluation and self._gate_iteration < 0 and self._gate.enabled[0]:
            self._gate_iteration = self._training_iteration
            print(f"Box termination enabled at collector iteration {self._gate_iteration}, "
                  f"episode steps EMA {self._average_episode_steps:.2f}", flush=True)


OFFICIAL_TASK = "g1-29dof-wbt-largebox-official325-critic-v2"


@registry.envcfg(OFFICIAL_TASK)
def make_official325_cfg() -> OmniRetargetEnvCfg:
    cfg = make_opentrack506_cfg()
    cfg.commands.motion.motion_files = (str(MOTION_DIR / "sub3_largebox_003_w_obj.npz"),)
    cfg.max_episode_seconds = 7.0
    cfg.object_curriculum.reference_frames = 325
    return cfg


registry.env(OFFICIAL_TASK)(OpenTrackCarryEnv)


NO_ROOT_POS_TASK = OFFICIAL_TASK + "-no-root-pos-noise"


@registry.envcfg(NO_ROOT_POS_TASK)
def make_official325_no_root_pos_noise_cfg() -> OmniRetargetEnvCfg:
    cfg = make_official325_cfg()
    cfg.sim_reset.body_pos.noise = (0.0, 0.0, 0.0)
    cfg.rewards.object_global_ref_position_error_exp.weight = 0.5
    cfg.rewards.object_global_ref_orientation_error_exp.weight = 0.5
    return cfg


registry.env(NO_ROOT_POS_TASK)(OpenTrackCarryEnv)
