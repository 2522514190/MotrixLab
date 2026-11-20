# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Run four cumulative WBT domain-randomization experiments sequentially."""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
BASE = "g1-29dof-wbt-largebox-opentrack674-loco-ik-end-sbto-omni-reward-joint-pos-delay01"
STAGES = ("mass", "mass-com", "mass-com-pd", "mass-com-pd-push")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "status.json").exists():
        raise FileExistsError(f"Queue already exists: {out}; use a new output directory.")
    status = {
        "pid": os.getpid(),
        "started": now(),
        "state": "starting",
        "iterations_per_stage": 100000,
        "num_envs": 2048,
        "seed": 2,
        "resume": None,
        "stages": [{"name": name, "env": f"{BASE}-{name}", "state": "pending"} for name in STAGES],
    }

    def save():
        status["updated"] = now()
        temporary = out / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(out / "status.json")

    # Capture sources before the first stage; keep the checkout unchanged until
    # all queued processes have imported these same implementations.
    for source in (
        "motrix_envs/src/motrix_envs/locomotion/wbt/g1/largebox.py",
        "motrix_env_core/src/motrix_env_core/sim/model.py",
        "motrix_env_motrixsim/src/motrix_env_motrixsim/runtime.py",
        "scripts/training/run_g1_wbt_progressive_dr.py",
    ):
        target = out / "source" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)
    from motrix_env_core import registry

    for stage in status["stages"]:
        cfg = registry.make_env_config(stage["env"])
        (out / f"{stage['name']}-env.json").write_text(json.dumps(asdict(cfg), indent=2, default=str) + "\n")
        recipe = Path("configs/task") / stage["env"] / "motrix.fastsac.yaml"
        target = out / "source" / recipe
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / recipe, target)
    save()

    for stage in status["stages"]:
        status["state"] = "running"
        stage.update(state="running", started=now(), iteration=0)
        task = f"{stage['env']}/motrix.fastsac"
        command = [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / ".venv/bin/mxlab"),
            "train",
            f"task={task}",
            "num_envs=2048",
            "seed=2",
            "algo.trainer.num_learning_iterations=100000",
            "logging.interval=100",
            "checkpoint.interval=1000",
        ]
        stage["command"] = command
        log_path = out / f"{stage['name']}.log"
        stage["log"] = str(log_path)
        save()
        print(f"{now()} START {stage['name']}", flush=True)
        process = None
        try:
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    start_new_session=True,
                )
                stage["child_pid"] = process.pid
                while True:
                    try:
                        code = process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        code = None
                    contents = log_path.read_text()
                    directory = re.search(r"created run context.*run_dir=([^\n)]+)", contents)
                    if directory:
                        stage["run_dir"] = str(ROOT / directory.group(1))
                    iterations = re.findall(r"iter (\d+)/100000", contents)
                    if iterations:
                        stage["iteration"] = int(iterations[-1])
                    save()
                    if code is not None:
                        break
            stage["exit_code"] = code
            if code:
                raise RuntimeError(f"Training exited with code {code}; see {log_path}")
            run_dir = Path(stage["run_dir"])
            metadata = json.loads((run_dir / "metadata.json").read_text())
            if metadata["performance"]["iterations"] != 100000:
                raise RuntimeError(f"Incomplete training: {run_dir}")
            checkpoint = run_dir / "checkpoints/latest.pt"
            if not checkpoint.is_file() or not checkpoint.stat().st_size:
                raise RuntimeError(f"Missing final checkpoint: {checkpoint}")
            shutil.copy2(log_path, run_dir / "training.log")
            shutil.copy2(out / f"{stage['name']}-env.json", run_dir / "environment_config.json")
            stage.update(state="complete", ended=now(), checkpoint=str(checkpoint), performance=metadata["performance"])
            save()
            print(f"{now()} COMPLETE {stage['name']} {run_dir}", flush=True)
        except BaseException as error:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            stage.update(state="failed", ended=now(), error=repr(error))
            status["state"] = "failed"
            save()
            raise
    status.update(state="complete", ended=now())
    save()
    print(f"{now()} ALL FOUR STAGES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
