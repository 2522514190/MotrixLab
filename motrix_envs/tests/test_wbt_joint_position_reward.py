# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

from motrix_envs.locomotion.wbt.mdp.rewards import joint_position_reward


@pytest.mark.parametrize("repeats", [1, 3])
def test_joint_position_reward_averages_actual_angle_errors_at_current_frame(repeats):
    reference = np.tile([[0.0, 0.0], [0.0, 2.0]], (1, repeats))
    actual = np.zeros(2 * repeats)
    motion = SimpleNamespace(steps=np.array([0]), clip=SimpleNamespace(joint_pos=reference))
    ctx = SimpleNamespace(commands={"motion": motion}, sim={"robot_dof_pos": actual})
    sigma = np.float32(2.0)

    assert joint_position_reward(ctx, sigma) == pytest.approx(1.0)
    motion.steps[0] = 1
    # Half the joints differ by two radians: mean squared error is two.
    assert joint_position_reward(ctx, sigma) == pytest.approx(np.exp(-0.5))
    actual[:] = reference[1]
    assert joint_position_reward(ctx, sigma) == pytest.approx(1.0)
    actual[:] += 1.0
    assert joint_position_reward(ctx, sigma) == pytest.approx(np.exp(-0.25))
