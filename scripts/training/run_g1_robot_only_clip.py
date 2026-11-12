"""Train a fresh 60k clipped policy, then evaluate and package it for PC2."""

import argparse
import json
import os
import shutil
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
MXLAB = ROOT / ".venv/bin/mxlab"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    status = dict(
        pid=os.getpid(),
        started=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        iterations=60000,
        num_envs=2048,
        seed=1,
        resume=None,
    )

    def save(stage, **fields):
        status.update(stage=stage, updated=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **fields)
        (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(status["updated"], stage, flush=True)

    def run(stage, command, extra_env=None):
        save(stage, command=[str(x) for x in command])
        with (out / (stage + ".log")).open("w") as log:
            process = subprocess.Popen(
                [str(x) for x in command],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=None if extra_env is None else {**os.environ, **extra_env},
            )
            save(stage, child_pid=process.pid)
            code = process.wait()
            if code:
                raise RuntimeError(f"{stage} exited with code {code}; see {out / (stage + '.log')}")

    try:
        parent = ROOT / "runs/g1-29dof-wbt-largebox/motrix/torch/fastsac"
        before = set(parent.glob("*"))
        run(
            "training",
            [
                PYTHON,
                MXLAB,
                "train",
                "task=g1-29dof-wbt-largebox/motrix.fastsac",
                "num_envs=2048",
                "seed=1",
                "algo.trainer.num_learning_iterations=60000",
                "checkpoint.interval=1000",
            ],
        )
        created = set(parent.glob("*")) - before
        if len(created) != 1:
            raise RuntimeError(f"expected one new training run, found {created}")
        run_dir = created.pop()
        checkpoint = run_dir / "checkpoints/latest.pt"
        save("trained", run_dir=str(run_dir), checkpoint=str(checkpoint))
        shutil.copytree(out / "source", run_dir / "source_snapshot")
        standard = out / "robot-only-clip.deploy"
        run(
            "export",
            [PYTHON, MXLAB, "export-deploy", f"run_dir={run_dir}", f"checkpoint={checkpoint}", f"output={standard}"],
        )
        for mode in ["nominal", "perturbed"]:
            command = [
                PYTHON,
                ROOT / "scripts/motion/evaluate_robot_only_clip.py",
                "--policy",
                checkpoint,
                "--output",
                out / f"evaluation-{mode}.json",
                "--trials",
                "20",
            ]
            if mode == "perturbed":
                command.append("--perturbations")
            run(f"evaluation-{mode}", command)
        bundle = out / "pc2-bundle"
        run(
            "package",
            [
                PYTHON,
                ROOT / "scripts/training/package_g1_robot_only_pc2.py",
                "--artifact",
                standard,
                "--output",
                bundle,
            ],
        )
        paths = [
            out / "pc2-runtime" / p / "src"
            for p in ["motrix_deploy", "motrix_deploy_tasks", "motrix_env_core", "motrix_robots"]
        ]
        paths.append(ROOT / "deployments/g1_pc2_compat/src")
        run(
            "pc2-offline-check",
            [
                PYTHON,
                ROOT / "scripts/training/check_g1_pc2_artifact.py",
                bundle / "artifacts/g1-robot-only-clip-pc2.deploy",
            ],
            {"PYTHONPATH": os.pathsep.join(map(str, paths))},
        )
        shutil.make_archive(str(out / "g1-robot-only-clip-pc2"), "gztar", bundle)
        save("complete", bundle=str(bundle), archive=str(out / "g1-robot-only-clip-pc2.tar.gz"))
    except BaseException:
        save("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
