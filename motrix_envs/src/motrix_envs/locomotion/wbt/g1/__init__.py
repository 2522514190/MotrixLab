# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unitree G1 whole-body tracking tasks.

Importing a task module registers its env config and env class; these
imports exist for that registration side effect.
"""

from . import backflip, dance, largebox, sbto005_teacher  # noqa: F401
from . import sbto005_student  # noqa: F401  registers the student task

__all__ = ["backflip", "dance", "largebox", "sbto005_teacher"]
