"""Evaluate complete robot-only carrying plus final hold without recovery resets."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from motrix_rl_builtin.fastsac.wrap_np import FastSacNpEnvWrap

import motrix_envs  # noqa: F401
from motrix_env_core import registry
from motrix_envs.locomotion.wbt.g1.largebox import RobotOnlyCarryEnv
from motrix_rl import runner, runs


class NoRecoveryEnv(RobotOnlyCarryEnv):
    def init_state(self):
        self._initializing = True
        try:
            return super().init_state()
        finally:
            self._initializing = False

    def _reset_done_envs(self):
        if self._initializing:
            super()._reset_done_envs()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--perturbations", action="store_true")
    args = parser.parse_args()
    found = runs.find_metadata_for_policy(args.policy)
    if found is None or args.trials < 1:
        parser.error("a metadata-backed checkpoint and positive trial count are required")
    run_dir, metadata = found
    if metadata.env_name != "g1-29dof-wbt-largebox":
        parser.error("expected the robot-only largebox task")
    trainer = runner.create_run_handle(
        runs.open_run_context(run_dir, metadata),
        cfg_override={"device": "cpu"},
        play_num_envs=1,
        seed=args.seed,
        render=None,
    ).trainer
    trainer._set_seed(args.seed)
    cfg = registry.make_env_config(metadata.env_name, mode="play")
    cfg.commands.motion.sequential_clips = False
    cfg.commands.motion.hold_at_clip_end = True
    cfg.randomization.enabled = args.perturbations
    # Full-start evaluation isolates observation noise and optional physical DR.
    for term in cfg.sim_reset.to_dict().values():
        term.noise_scale = 0.0
    inner = NoRecoveryEnv(cfg, num_envs=1, backend=metadata.sim, seed=args.seed)
    env = FastSacNpEnvWrap(inner, torch.device("cpu"))
    agent = trainer._make_agent(env)
    agent.load_state_dict(torch.load(args.policy, map_location="cpu", weights_only=False))
    agent.actor.eval()
    agent.obs_normalizer.eval()
    episodes = []
    for trial in range(args.trials):
        obs, _ = env.reset()
        motion = inner.command_terms["motion"]
        term = inner.action_terms["joint_position"]
        frames = len(motion.clip.joint_pos)
        steps = frames + round(5.0 / cfg.ctrl_dt)
        error_sq = 0.0
        clipped = violated = total = 0
        max_violation = 0.0
        reasons = []
        for tick in range(steps):
            frame = int(motion.steps[0, 0])
            with torch.inference_mode():
                actions = agent.act(obs, deterministic=True)
            raw = actions.cpu().numpy()
            target = raw * term.state.action_scales + term.state.default_angles
            lo, hi = term._target_limits
            clipped += int(np.count_nonzero((target < lo) | (target > hi)))
            result = env.step(actions)
            obs = result.obs
            q = inner.sim_data["robot_dof_pos"][0]
            error_sq += float(np.sum((q - motion.clip.joint_pos[frame]) ** 2))
            violation = np.maximum(term.state.joint_lower - q, 0) + np.maximum(q - term.state.joint_upper, 0)
            max_violation = max(max_violation, float(np.max(violation)))
            violated += int(np.count_nonzero(violation > 0))
            total += q.size
            if bool(result.terminated.any()) or bool(result.truncated.any()):
                reasons = [name for name in cfg.terminations.to_dict() if np.any(inner.state.metrics[name])]
                reasons = reasons or ["truncated"]
                break
        entry = dict(
            trial=trial,
            success=not reasons,
            steps=tick + 1,
            reference_frame=frame,
            reasons=reasons,
            joint_tracking_rmse_rad=float(np.sqrt(error_sq / total)),
            target_clip_fraction=clipped / total,
            actual_joint_violation_fraction=violated / total,
            max_actual_joint_violation_rad=max_violation,
        )
        episodes.append(entry)
        print(json.dumps(entry), flush=True)
    report = dict(
        policy=str(args.policy.resolve()),
        perturbations=args.perturbations,
        push_enabled=cfg.randomization.push_enabled,
        seed=args.seed,
        hold_s=5.0,
        trials=args.trials,
        successes=sum(e["success"] for e in episodes),
        success_rate=sum(e["success"] for e in episodes) / args.trials,
        criterion="full clip from frame zero plus 5s final hold without termination or recovery",
        observation_noise=True,
        reset_noise=False,
        failure_counts=dict(Counter(r for e in episodes for r in e["reasons"])),
        episodes=episodes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    env.close()


if __name__ == "__main__":
    main()
