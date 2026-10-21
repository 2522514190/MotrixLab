# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0

"""OmniRetarget task with Holosoma's finite FastSAC policy-action bounds."""

import numpy as np

from motrix_env_core import registry
from motrix_env_core.mdp.action_space import symmetric_residual_action_space
from motrix_envs.locomotion.wbt.g1.omniretarget import (
    OmniRetargetEnv,
    OmniRetargetEnvCfg,
    make_omniretarget_cfg,
)


@registry.envcfg("g1-29dof-wbt-largebox-omniretarget-fastsac")
def make_omniretarget_fastsac_cfg() -> OmniRetargetEnvCfg:
    """Reuse the PPO task configuration, including its unbounded PD setpoints.

    zh_CN: 复用 PPO 搬箱配置，仅由任务运行时提供 FastSAC 的有限策略动作边界。
    """
    return make_omniretarget_cfg()


@registry.env("g1-29dof-wbt-largebox-omniretarget-fastsac")
class OmniRetargetFastSacEnv(OmniRetargetEnv):
    """Bound policy output without changing the shared calibrated PD mapping.

    Holosoma ``FastSACEnvWrapper._compute_action_boundaries`` uses the largest
    distance from the default pose to a mechanical joint limit, divided by the
    scalar robot ``control.action_scale``. Preserve that scalar convention;
    the inherited action term still applies the per-joint effort/kp scales.
    """

    def __init__(self, cfg: OmniRetargetEnvCfg, num_envs=1, backend=None, seed=None):
        super().__init__(cfg, num_envs=num_envs, backend=backend, seed=seed)
        action = self.action_terms["joint_position"].state
        self._action_space = symmetric_residual_action_space(
            np.stack((action.joint_lower, action.joint_upper)),
            action.default_angles,
            cfg.actions.joint_position.action_scale,
        )
