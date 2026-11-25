# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""FastSAC distribution head for the distilled SBTO005 policy."""

import torch
from torch import nn

from motrix_envs.locomotion.wbt.g1.sbto005_distill.camera import HEIGHT, WIDTH
from motrix_envs.locomotion.wbt.g1.sbto005_distill.policy import ACTION_SIZE, STATE_SIZE, Student
from motrix_rl_builtin.fastsac.networks import Actor

OBS_SIZE = STATE_SIZE + HEIGHT * WIDTH


class Sbto005Actor(Actor):
    def __init__(self, cfg, n_obs, n_act, action_scale, action_bias, device):
        nn.Module.__init__(self)
        if (n_obs, n_act) != (OBS_SIZE, ACTION_SIZE) or not cfg.use_tanh:
            raise ValueError("The SBTO005 depth actor requires 3226 observations and 29 tanh-squashed actions.")
        self.n_act = n_act
        self.log_std_max = cfg.log_std_max
        self.log_std_min = cfg.log_std_min
        self.use_tanh = True
        scale = torch.ones(n_act) if action_scale is None else action_scale
        bias = torch.zeros(n_act) if action_bias is None else action_bias
        self.student = Student(bias - scale, bias + scale)
        self.fc_logstd = nn.Linear(self.student.mlp[6].in_features, n_act)
        nn.init.zeros_(self.fc_logstd.weight)
        nn.init.zeros_(self.fc_logstd.bias)
        self.to(device)

    @property
    def action_scale(self):
        return self.student.half_range

    @property
    def action_bias(self):
        return self.student.center

    def initialize_from_distillation(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["student"]
        for name, expected in (("center", self.action_bias), ("half_range", self.action_scale)):
            torch.testing.assert_close(state[name], expected.cpu(), rtol=0, atol=1e-6)
        self.student.load_state_dict(state, strict=True)
        
        # Handle both old (global log_std) and new (conditional fc_logstd) formats
        if "fc_logstd" in checkpoint:
            # New format: directly load the conditional log_std head
            self.fc_logstd.load_state_dict(checkpoint["fc_logstd"])
        elif "log_std" in checkpoint:
            # Legacy format: map global log_std to fc_logstd initialization
            # (This provides warm-start for RL but RL will adapt it)
            with torch.no_grad():
                log_std = checkpoint["log_std"].to(self.fc_logstd.bias)
                fraction = (log_std - self.log_std_min) / (self.log_std_max - self.log_std_min)
                if not bool(((fraction >= 0) & (fraction <= 1)).all()):
                    raise ValueError("Distillation log_std lies outside the FastSAC actor range.")
                self.fc_logstd.weight.zero_()
                self.fc_logstd.bias.copy_((2 * fraction - 1).clamp(-1 + 1e-6, 1 - 1e-6).atanh())

    def forward(self, obs):
        state = obs[:, :STATE_SIZE]
        depth = obs[:, STATE_SIZE:].reshape(-1, 1, HEIGHT, WIDTH)
        features = self.student.features(state, depth)
        mean = self.student.mlp[6](features)
        log_std_fraction = (self.fc_logstd(features).tanh() + 1) / 2
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * log_std_fraction
        action = mean.tanh() * self.action_scale + self.action_bias
        return action, mean, log_std
