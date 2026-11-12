# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Deployment profile provider for built-in environments."""


def register() -> None:
    """Register the built-in environment-to-profile compiler mappings."""
    from motrix_deploy.profile import register_profile_compiler
    from motrix_envs.deploy.g1_walk import build_g1_walk_profile
    from motrix_envs.deploy.g1_wbt import build_g1_wbt_profile
    from motrix_envs.deploy.go2_walk import build_go2_walk_profile

    register_profile_compiler("go2-walk-flat", build_go2_walk_profile)
    register_profile_compiler("go2-walk-rough", build_go2_walk_profile)
    register_profile_compiler("g1-wbt-dance", build_g1_wbt_profile)
    register_profile_compiler("g1-29dof-wbt-largebox", build_g1_wbt_profile)
    for suffix in (
        "",
        *(f"-{letter}" for letter in "abcdefgh"),
        "-a-ground-action-baseline",
        "-a-ground-action-smooth01",
        "-a-ground-action-dr05",
        "-a-ground-action-dr01",
    ):
        register_profile_compiler(f"g1-29dof-wbt-largebox-omniretarget-fastsac{suffix}", build_g1_wbt_profile)
    register_profile_compiler("g1-walk-flat", build_g1_walk_profile)
