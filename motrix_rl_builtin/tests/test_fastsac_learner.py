# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Update-gating and timing-helpers tests for the FastSAC agent/learner."""

from types import SimpleNamespace


def _tiny_agent_cfg(**overrides):
    """Minimal valid FastSacAgentCfg namespace for a real (tiny) CUDA agent."""
    values = dict(
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        num_q_networks=2,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        alpha_learning_rate=1e-3,
        override_resume_learning_rates=False,
        weight_decay=0.0,
        max_grad_norm=0.0,
        use_layer_norm=False,
        use_tanh=True,
        log_std_max=0.0,
        log_std_min=-5.0,
        num_atoms=5,
        v_min=-20.0,
        v_max=20.0,
        gamma=0.97,
        tau=0.125,
        alpha_init=0.001,
        use_autotune=True,
        target_entropy_ratio=0.0,
        buffer_size=64,
        num_steps=1,
        batch_size=8,
        learning_starts=1,
        policy_frequency=4,
        num_updates=4,
        obs_normalization=True,
        compile=True,
        amp=True,
        amp_dtype="bf16",
        device=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_resume_can_override_learning_rates_while_restoring_optimizer_state() -> None:
    import torch
    from motrix_rl_builtin.fastsac.agent import FastSacAgent

    dims = dict(obs_dim=5, critic_obs_dim=7, act_dim=3, num_envs=8, device=torch.device("cpu"))
    source = FastSacAgent(**dims, cfg=_tiny_agent_cfg())
    source.global_step = 42
    names = ("actor_optimizer", "q_optimizer", "alpha_optimizer")
    for name in names:
        optimizer = getattr(source, name)
        param = optimizer.param_groups[0]["params"][0]
        param.grad = torch.ones_like(param)
        optimizer.step()
    checkpoint = source.state_dict()
    rates = (1e-4, 2e-4, 3e-4)

    for override in (False, True):
        resumed = FastSacAgent(
            **dims,
            cfg=_tiny_agent_cfg(
                actor_learning_rate=rates[0],
                critic_learning_rate=rates[1],
                alpha_learning_rate=rates[2],
                override_resume_learning_rates=override,
            ),
        )
        resumed.load_state_dict(checkpoint)
        assert resumed.global_step == source.global_step
        for name, rate in zip(names, rates, strict=True):
            state = resumed.state_dict()[name]
            assert state["param_groups"][0]["lr"] == (rate if override else checkpoint[name]["param_groups"][0]["lr"])
            torch.testing.assert_close(state["state"], checkpoint[name]["state"])


def test_policy_frequency_gating_spans_update_calls() -> None:
    """Actor updates exactly every policy_frequency-th gradient step, across calls.

    Gating reads the persistent ``update_idx`` counter, so the actor/Q ratio is
    exactly ``1 / policy_frequency`` across update() calls, not just within one
    (e.g. frequency 2: 4 steps -> 2 actor calls, 3 steps -> 1).
    """
    import torch
    from motrix_rl_builtin.fastsac.agent import FastSacAgent

    n_env, obs, cri, act = 8, 5, 7, 3
    agent = FastSacAgent(
        obs_dim=obs,
        critic_obs_dim=cri,
        act_dim=act,
        num_envs=n_env,
        cfg=_tiny_agent_cfg(policy_frequency=2, amp=False, compile=False),
        device=torch.device("cpu"),
    )
    for _ in range(4):
        agent.rb.extend(
            torch.randn(n_env, obs),
            torch.randn(n_env, cri),
            torch.randn(n_env, act),
            torch.randn(n_env),
            torch.zeros(n_env, dtype=torch.long),
            torch.zeros(n_env, dtype=torch.long),
        )
    agent._update_main = lambda b: None
    actor_calls: list[int] = []
    agent._update_pol = lambda b: actor_calls.append(1)

    agent.update(4)  # update_idx 0 -> 4: actor gates at steps 0, 2
    assert len(actor_calls) == 2
    agent.update(4)  # update_idx 4 -> 8: actor gates at steps 4, 6
    assert len(actor_calls) == 4
    agent.update(3)  # update_idx 8 -> 11: actor gates at steps 8, 10
    assert len(actor_calls) == 6
    assert agent.update_idx == 11


def test_nest_timing_path_merges_scalar_total_with_children_in_any_order() -> None:
    from motrix_rl_builtin.fastsac.async_impl.stats import nest_timing_path

    # the collector emits a stage's scalar before its dotted sub-stages; the
    # rebuild must fold it into a "total" instead of nesting under a float
    tree: dict = {}
    nest_timing_path(tree, ("physics",), 15.0)
    nest_timing_path(tree, ("physics", "read"), 12.0)
    assert tree == {"physics": {"total": 15.0, "read": 12.0}}

    # reverse order must converge to the same tree
    tree = {}
    nest_timing_path(tree, ("physics", "read"), 12.0)
    nest_timing_path(tree, ("physics",), 15.0)
    assert tree == {"physics": {"total": 15.0, "read": 12.0}}

    # stages without sub-stages stay scalar
    tree = {}
    nest_timing_path(tree, ("apply_action",), 1.5)
    assert tree == {"apply_action": 1.5}
