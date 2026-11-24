# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Depth-conditioned deterministic SBTO005 policy."""

import torch
from torch import nn

STATE_SIZE = 154
ACTION_SIZE = 29
STATE_TERMS = ("motion_joint", "motion_ref_ori_b", "base_ang_vel", "dof_pos", "dof_vel", "actions")


class Student(nn.Module):
    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        super().__init__()
        self.register_buffer("center", (high + low) / 2)
        self.register_buffer("half_range", (high - low) / 2)
        self.register_buffer("state_mean", torch.zeros(STATE_SIZE))
        self.register_buffer("state_scale", torch.ones(STATE_SIZE))
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2),
            nn.ELU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ELU(),
            nn.Conv2d(32, 32, 3),
            nn.ELU(),
        )
        y, x = torch.meshgrid(torch.linspace(-1.0, 1.0, 8), torch.linspace(-1.0, 1.0, 12), indexing="ij")
        self.register_buffer("coordinates", torch.stack((x.flatten(), y.flatten()), dim=-1))
        self.mlp = nn.Sequential(
            nn.Linear(STATE_SIZE + 64, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, ACTION_SIZE),
            nn.Tanh(),
        )

    def features(self, state: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        state = ((state - self.state_mean) / self.state_scale).clamp(-10, 10)
        features = self.cnn(depth).flatten(start_dim=2).softmax(dim=-1)
        keypoints = (features @ self.coordinates).flatten(start_dim=1)
        x = torch.cat((state, keypoints), dim=-1)
        for layer in list(self.mlp)[:6]:
            x = layer(x)
        return x

    def forward(self, state: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return self.mlp[7](self.mlp[6](self.features(state, depth))) * self.half_range + self.center


def student_terms(env) -> torch.Tensor:
    layout = env.manager_layout.observations["policy"]
    slices = {term.name: term.output_slice for term in layout.terms}
    columns = torch.cat([torch.arange(slices[name].start, slices[name].stop) for name in STATE_TERMS])
    if columns.numel() != STATE_SIZE or env.observation_manager.spec.policy.space.shape != (172,):
        raise ValueError("Unexpected SBTO005 Teacher/Student observation layout.")
    return columns
