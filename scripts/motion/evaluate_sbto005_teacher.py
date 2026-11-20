# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a metadata-backed SBTO 005 teacher on uninterrupted frame-zero trials."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from motrix_rl_builtin.fastsac.wrap_np import FastSacNpEnvWrap

from motrix_env_core import registry
from motrix_envs.locomotion.wbt.g1.sbto005_teacher import TASK, Sbto005TeacherEnv
from motrix_rl import runner, runs


class EvaluationEnv(Sbto005TeacherEnv):
    """Keep the task's terminal simulator state available for scoring."""

    _initializing = False

    def init_state(self):
        self._initializing = True
        try:
            return super().init_state()
        finally:
            self._initializing = False

    def _reset_done_envs(self) -> None:
        if self._initializing:
            super()._reset_done_envs()


def evaluate_episode(env: FastSacNpEnvWrap, act: Callable[[torch.Tensor], torch.Tensor]) -> dict:
    obs, _ = env.reset()
    inner = env.env
    motion = inner.command_terms["motion"]
    ref_position = motion.clip.extensions["object_pos_w"].data
    ref_orientation = motion.clip.extensions["object_quat_w"].data
    start_height = float(ref_position[0, 2])
    peak_lift = 0.0
    lift_streak = 0
    setdown_streak = 0
    lifted = False
    set_down = False
    max_error = 0.0

    for frame in range(len(ref_position)):
        with torch.inference_mode():
            result = env.step(act(obs))
        obs = result.obs
        position = inner.sim_data["object_pos"][0]
        velocity = inner.sim_data["object_lin_vel"][0]
        error = float(np.linalg.norm(position - ref_position[frame]))
        max_error = max(max_error, error)
        height = float(position[2]) - start_height
        peak_lift = max(peak_lift, height)
        if not lifted:
            lift_now = height >= 0.25 and ref_position[frame, 2] - start_height >= 0.25 and error <= 0.20
            lift_streak = lift_streak + 1 if lift_now else 0
            lifted = lift_streak >= 25
        elif not set_down:
            setdown_now = (
                ref_position[frame, 2] - start_height <= 0.08
                and abs(height) <= 0.08
                and error <= 0.20
                and np.linalg.norm(velocity) <= 0.20
            )
            setdown_streak = setdown_streak + 1 if setdown_now else 0
            set_down = setdown_streak >= 15
        if result.terminated[0] or result.truncated[0]:
            break

    completed = bool(inner.state.metrics["episode_clip_end"][0]) and not bool(result.terminated[0])
    final_pos_error = float(np.linalg.norm(position - ref_position[-1]))
    orientation = inner.sim_data["object_quat"][0]
    quat_dot = abs(float(np.dot(orientation, ref_orientation[-1]))) / (
        np.linalg.norm(orientation) * np.linalg.norm(ref_orientation[-1])
    )
    final_angle_error = float(2 * np.arccos(np.clip(quat_dot, 0.0, 1.0)))
    return {
        "steps": frame + 1,
        "completed_clip": completed,
        "lifted": lifted,
        "set_down": set_down,
        "success": completed and lifted and set_down and final_pos_error < 0.15 and final_angle_error < 0.30,
        "peak_lift_m": peak_lift,
        "max_tracking_error_m": max_error,
        "final_position_error_m": final_pos_error,
        "final_angle_error_rad": final_angle_error,
        "failure_terms": [name for name in inner.termination_manager.terms if inner.state.metrics[name][0]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", type=Path, required=True, help="FastSAC checkpoint from a metadata-backed Teacher run"
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--training-perturbations", action="store_true", help="Apply training DR and noise while fixing frame zero"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    found = runs.find_metadata_for_policy(args.policy)
    if found is None:
        parser.error("the checkpoint must be part of a metadata-backed run")
    run_dir, metadata = found
    if (metadata.env_name, metadata.rllib, metadata.algo) != (TASK, "motrix", "fastsac"):
        parser.error(f"expected a Motrix FastSAC checkpoint for {TASK}")

    run = runs.open_run_context(run_dir, metadata)
    trainer = runner.create_run_handle(
        run, cfg_override={"device": args.device}, play_num_envs=1, seed=args.seed
    ).trainer
    cfg = registry.make_env_config(TASK, mode="train" if args.training_perturbations else "play")
    cfg.commands.motion.adaptive_sampling_enabled = False
    cfg.commands.motion.start_at_timestep_zero_prob = 1.0
    inner = EvaluationEnv(cfg, num_envs=1, backend=metadata.sim, seed=args.seed)
    env = FastSacNpEnvWrap(inner, torch.device(args.device))
    try:
        agent = trainer._make_agent(env)
        checkpoint = torch.load(args.policy, map_location=args.device, weights_only=True)
        agent.load_state_dict(checkpoint, load_optimizers=False)
        agent.actor.eval()
        agent.obs_normalizer.eval()
        results = [evaluate_episode(env, lambda obs: agent.act(obs, deterministic=True)) for _ in range(args.trials)]
    finally:
        env.close()
    report = {
        "task": TASK,
        "policy": str(args.policy.resolve()),
        "seed": args.seed,
        "training_perturbations": args.training_perturbations,
        "completed_clip_rate": sum(trial["completed_clip"] for trial in results) / len(results),
        "carry_success_rate": sum(trial["success"] for trial in results) / len(results),
        "trials": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
