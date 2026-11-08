# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared nominal OmniRetarget carrying world for all FastSAC ablations."""

from motrix_env_core.config.scene import SceneCfg
from motrix_envs.locomotion.wbt.g1.omniretarget import make_omniretarget_cfg


def build_scene() -> SceneCfg:
    scene = make_omniretarget_cfg().for_play().scene
    scene.objs.robot = None
    return scene
