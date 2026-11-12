# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0

"""Four independent A-baseline ground-friction/action-rate experiments."""

import numpy as np

from motrix_env_core import registry
from motrix_env_core.sim.model import GeomFrictionQuery
from motrix_env_core.sim.write import GeomFrictionWrite
from motrix_envs.locomotion.wbt.g1.omniretarget import OmniRetargetEnvCfg
from motrix_envs.locomotion.wbt.g1.omniretarget_fastsac import OmniRetargetFastSacEnv, make_a_cfg

PREFIX = "g1-29dof-wbt-largebox-omniretarget-fastsac-"


def _make_cfg(weight: float, ground_dr: bool) -> OmniRetargetEnvCfg:
    cfg = make_a_cfg()
    cfg.rewards.action_rate_l2.weight = weight
    if ground_dr:
        cfg.scene.objs.floor.priority = 2
        cfg.queries.model["ground_action_friction"] = GeomFrictionQuery(name="floor")
    return cfg


@registry.envcfg(PREFIX + "a-ground-action-baseline")
def make_baseline_cfg() -> OmniRetargetEnvCfg:
    return _make_cfg(-0.5, False)


@registry.envcfg(PREFIX + "a-ground-action-smooth01")
def make_smooth01_cfg() -> OmniRetargetEnvCfg:
    return _make_cfg(-0.1, False)


@registry.envcfg(PREFIX + "a-ground-action-dr05")
def make_dr05_cfg() -> OmniRetargetEnvCfg:
    return _make_cfg(-0.5, True)


@registry.envcfg(PREFIX + "a-ground-action-dr01")
def make_dr01_cfg() -> OmniRetargetEnvCfg:
    return _make_cfg(-0.1, True)


class GroundActionDrEnv(OmniRetargetFastSacEnv):
    """Retain one ground-friction draw per lane through all native resets."""

    def __init__(self, cfg, num_envs=1, backend=None, seed=None):
        super().__init__(cfg, num_envs=num_envs, backend=backend, seed=seed)
        self._ground_writes = self.sim.compile_writes({"friction": GeomFrictionWrite(("floor",))}, reset=False)
        friction = self._ground_writes.buffer("friction")
        friction[:, 0] = self.model.others["ground_action_friction"]
        rng = np.random.default_rng(seed)
        friction[:, 0, 0] = (
            cfg.play_ground_friction
            if cfg.evaluation and cfg.play_ground_friction is not None
            else rng.uniform(0.3, 1.6, size=num_envs)
        )

    def reset(self, env_ids, sim_reset_ids=None):
        super().reset(env_ids, sim_reset_ids)
        selected = env_ids if sim_reset_ids is None else np.union1d(env_ids, sim_reset_ids)
        if selected.size:
            self._ground_writes.execute(selected)


for suffix in ("a-ground-action-baseline", "a-ground-action-smooth01"):
    registry.env(PREFIX + suffix)(OmniRetargetFastSacEnv)
for suffix in ("a-ground-action-dr05", "a-ground-action-dr01"):
    registry.env(PREFIX + suffix)(GroundActionDrEnv)
