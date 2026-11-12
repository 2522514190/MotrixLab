# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from motrix_rl import runs
from motrix_rl.commands import play
from motrix_rl.config import PlayConfig


def _metadata_run(tmp_path: Path, env_name: str = "go2", sim: str | None = None) -> Path:
    run_dir = tmp_path / "run"
    policy_path = run_dir / "checkpoints" / "best.pt"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text("checkpoint", encoding="utf-8")
    runs.write_metadata(
        run_dir,
        runs.make_metadata(env_name, "rslrl", "torch", "ppo", 1, "pt", sim=sim),
    )
    return policy_path


def test_policy_target_uses_metadata_when_optional_overrides_are_unset(tmp_path):
    policy_path = _metadata_run(tmp_path)

    target = play._policy_play_target(policy_path, None)

    assert target.run.metadata.env_name == "go2"
    assert target.run.metadata.sim is None


def test_policy_target_validates_explicit_env_and_overrides_sim(tmp_path):
    policy_path = _metadata_run(tmp_path)

    with pytest.raises(ValueError, match="does not match policy metadata"):
        play._policy_play_target(policy_path, "cartpole")

    target = play._policy_play_target(policy_path, "go2", sim="motrixsim")
    assert target.run.metadata.sim == "motrixsim"


def test_default_play_target_uses_cartpole(monkeypatch, tmp_path):
    policy_path = _metadata_run(tmp_path, env_name="cartpole")
    expected = play._policy_play_target(policy_path, None)
    requested = []

    def latest_target(env_name: str, rllib: str | None = None):
        requested.append((env_name, rllib))
        return expected

    monkeypatch.setattr(play, "_latest_play_target", latest_target)

    assert play._resolve_play_target(PlayConfig()) == expected
    assert requested == [("cartpole", None)]


@pytest.mark.parametrize("env_name", ["g1-29dof-wbt-largebox", "g1-29dof-wbt-largebox-omniretarget-fastsac-a"])
@pytest.mark.parametrize("perturbations", [True, False])
def test_carry_play_forwards_perturbation_controls(monkeypatch, tmp_path, env_name, perturbations):
    """Both carrying policies expose the same play perturbation controls."""
    from types import SimpleNamespace

    policy_path = _metadata_run(tmp_path, env_name=env_name)
    recorded = {}
    monkeypatch.setattr(play, "_play_render_config", lambda target, cfg: None)

    def create_handle(run, **kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(play=lambda path: recorded.update(policy_path=path))

    monkeypatch.setattr(play.runner, "create_run_handle", create_handle)
    play.run(PlayConfig(policy=str(policy_path), perturbations=perturbations, ground_friction=0.9))
    assert recorded["env_cfg_override"] == {
        "play_training_perturbations": perturbations,
        "play_ground_friction": 0.9,
    }
    assert recorded["policy_path"] == str(policy_path)
