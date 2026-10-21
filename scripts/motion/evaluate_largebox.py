# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Evaluate robot-only or physical-box FastSAC tracking without recovery resets."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from motrix_rl_builtin.fastsac.wrap_np import FastSacNpEnvWrap

import motrix_envs  # noqa: F401 registers built-in environments
from motrix_env_core import registry
from motrix_env_core.manager import ManagerEnv
from motrix_rl import runner, runs


class NoRecoveryEnv(ManagerEnv):
    """Retain the terminal physical state until the evaluator ends the trial."""

    def init_state(self):
        self._initializing = True
        try:
            return super().init_state()
        finally:
            self._initializing = False

    def _reset_done_envs(self) -> None:
        if self._initializing:
            super()._reset_done_envs()


def evaluate_episode(env, agent, num_frames: int) -> dict:
    obs, _ = env.reset()
    motion = env.env.command_terms["motion"]
    for step in range(1, num_frames + 1):
        target_frame = int(motion.steps[0, 0])
        with torch.inference_mode():
            actions = agent.act(obs, deterministic=True)
        obs, _, _, terminated, truncated = env.step(actions)
        # Failure takes precedence even on the final target frame.
        if bool(terminated.any()) or bool(truncated.any()):
            metrics = env.env._state.metrics
            reasons = [name for name in env.env.cfg.terminations.to_dict() if np.any(metrics[name])]
            result = {
                "success": False,
                "steps": step,
                "frame": target_frame,
                "reasons": reasons or ["truncated"],
            }
            if "object_position_error" in metrics:
                result["object_position_error_m"] = float(metrics["object_position_error"].item())
                result["object_orientation_error_rad"] = float(metrics["object_orientation_error"].item())
            return result
    return {"success": True, "steps": num_frames, "frame": num_frames - 1, "reasons": []}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mass", type=float, default=1.0, help="Fixed evaluation box mass in kg")
    parser.add_argument(
        "--env",
        choices=("g1-29dof-wbt-largebox", "g1-29dof-wbt-largebox-object"),
        help="Override the evaluation task; use the object task for older physical-box checkpoints",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.mass) or args.mass <= 0.0:
        parser.error("--mass must be finite and positive")
    if args.trials < 1:
        parser.error("--trials must be positive")
    found = runs.find_metadata_for_policy(args.policy)
    if found is None:
        parser.error("policy must belong to a metadata-backed training run")
    run_dir, metadata = found
    supported_envs = ("g1-29dof-wbt-largebox", "g1-29dof-wbt-largebox-object")
    if metadata.env_name not in supported_envs or (metadata.rllib, metadata.algo) != ("motrix", "fastsac"):
        parser.error("this evaluator supports the largebox robot-only and object Motrix FastSAC tasks")
    env_name = args.env or metadata.env_name
    run = runs.open_run_context(run_dir, metadata)
    trainer = runner.create_run_handle(
        run, cfg_override={"device": args.device}, play_num_envs=1, seed=args.seed, render=None
    ).trainer
    trainer._set_seed(args.seed)
    cfg = registry.make_env_config(env_name, mode="play")
    # Reset always selects zero; reaching the final frame never teleports the scene.
    with_object = env_name == "g1-29dof-wbt-largebox-object"
    if with_object:
        cfg.sim_reset.object_mass.light_range = (args.mass, args.mass)
        cfg.sim_reset.object_mass.heavy_range = (args.mass, args.mass)
    cfg.commands.motion.sequential_clips = False
    cfg.commands.motion.hold_at_clip_end = True
    inner = NoRecoveryEnv(cfg, num_envs=1, backend=metadata.sim, seed=args.seed)
    env = FastSacNpEnvWrap(inner, torch.device(args.device))
    try:
        agent = trainer._make_agent(env)
        agent.load_state_dict(torch.load(args.policy, map_location=args.device, weights_only=False))
        agent.actor.eval()
        agent.obs_normalizer.eval()
        env.reset()
        motion = inner.command_terms["motion"]
        num_frames = motion.clip.joint_pos.shape[0]
        if np.unique(motion.clip.frame_clip_end).size != 1:
            parser.error("evaluation requires a single reference clip")
        results = []
        for trial in range(args.trials):
            result = evaluate_episode(env, agent, num_frames)
            results.append(result)
            print(f"Trial {trial + 1}/{args.trials}: {json.dumps(result)}", flush=True)
        successes = sum(result["success"] for result in results)
        report = {
            "policy": str(args.policy.resolve()),
            "env_name": env_name,
            "seed": args.seed,
            "box_mass_kg": args.mass if with_object else None,
            "trials": args.trials,
            "successes": successes,
            "success_rate": successes / args.trials,
            "criterion": "reach the final reference frame from frame zero without any termination or recovery",
            "conditions": "current repository assets; play reset noise disabled; policy observation noise retained",
            "failure_counts": dict(Counter(reason for result in results for reason in result["reasons"])),
            "episodes": results,
        }
        output = args.output or run_dir / "evaluation_complete.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Complete success rate: {successes}/{args.trials} = {successes / args.trials:.1%}")
        print(f"Report: {output.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
