# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Train and evaluate the Gaussian SBTO005 depth Student."""

import argparse
import json
import os
from pathlib import Path

import torch

from motrix_envs.locomotion.wbt.g1.sbto005_distill.evaluation import evaluate_unfiltered
from motrix_envs.locomotion.wbt.g1.sbto005_distill.gaussian_training import GaussianTrainingSpec, train_gaussian
from motrix_envs.locomotion.wbt.g1.sbto005_distill.rollout import configure_motion
from motrix_envs.locomotion.wbt.g1.sbto005_distill.training import sha256
from motrix_envs.locomotion.wbt.g1.sbto005_teacher import TASK


def cloud(overrides: list[str]) -> GaussianTrainingSpec:
    opts = dict(arg.split("=", 1) for arg in overrides)
    if opts.pop("task") != TASK + "/motrix.fastsac" or opts.pop("task.train_backend") != "torch":
        raise ValueError("Expected the SBTO005 Teacher task in the Pipeline experiment.")
    for key in ("render", "play"):
        if opts.pop(key, "false") != "false":
            raise ValueError("Cloud distillation runs headless.")
    for key in ("checkpoint.interval", "logging.interval", "logging.backend"):
        opts.pop(key, None)
    envs = int(opts.pop("num_envs", "4096"))
    seed = int(opts.pop("seed", "2"))
    if envs != 4096 or opts:
        raise ValueError(f"Expected 4096 environments and supported Pipeline options, got {envs=} {opts=}.")
    teacher = Path(os.environ["MOTRIX_DISTILL_TEACHER"]).resolve()
    motion = Path(os.environ["MOTRIX_SBTO005_MOTION"]).resolve()
    for path, key in ((teacher, "TEACHER"), (motion, "MOTION")):
        if sha256(path) != os.environ[f"MOTRIX_DISTILL_{key}_SHA256"]:
            raise ValueError(f"Checksum mismatch for {path}.")
    rounds = int(os.environ["MOTRIX_DISTILL_ROUNDS"])
    return GaussianTrainingSpec(
        teacher,
        motion,
        Path("runs/sbto005-distill/gaussian").absolute(),
        envs=envs,
        seed=seed,
        learners=int(os.environ["MOTRIX_DISTILL_LEARNERS"]),
        rounds=rounds,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    train = commands.add_parser("local")
    train.add_argument("--teacher", type=Path, required=True)
    train.add_argument("--motion", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--envs", type=int, default=64)
    train.add_argument("--learners", type=int, default=1)
    train.add_argument("--rounds", type=int, default=10)
    train.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--motion", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--trials", type=int, default=200)
    evaluate.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    commands.add_parser("cloud").add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.mode == "cloud":
        train_gaussian(cloud(args.overrides))
    elif args.mode == "local":
        train_gaussian(
            GaussianTrainingSpec(
                args.teacher.resolve(),
                args.motion.resolve(),
                args.out.absolute(),
                envs=args.envs,
                learners=args.learners,
                rounds=args.rounds,
                device=args.device,
            )
        )
    else:
        configure_motion(args.motion)
        report = evaluate_unfiltered(args.checkpoint, args.trials, torch.device(args.device))
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
