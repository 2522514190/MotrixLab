"""Run the four 100k A-group experiments sequentially in a persistent host."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "g1-29dof-wbt-largebox-omniretarget-fastsac-"
GROUPS = [
    ("a-ground-action-baseline", -0.5, "1.0"),
    ("a-ground-action-smooth01", -0.1, "1.0"),
    ("a-ground-action-dr05", -0.5, "U(0.3,1.6)"),
    ("a-ground-action-dr01", -0.1, "U(0.3,1.6)"),
]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def wait_for_gpu(gpu):
    """Wait for existing compute processes, including other training workers."""
    while True:
        result = subprocess.run(
            ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        training_pids = []
        for process in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                args = process.read_bytes().split(b"\0")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if b"train" in args and any(Path(os.fsdecode(arg)).name == "mxlab" for arg in args if arg):
                training_pids.append(process.parent.name)
        if not result.stdout.strip() and not training_pids:
            return
        print(now(), "Waiting for GPU compute processes", result.stdout.strip(), training_pids, flush=True)
        time.sleep(60)


def run_queue(out, gpu):
    python = str(ROOT / ".venv/bin/python")
    mxlab = str(ROOT / ".venv/bin/mxlab")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_json(out / "queue.json", dict(start=now(), commit=commit, gpu=gpu, groups=GROUPS))
    snapshot = out / "source"
    snapshot.mkdir()
    sources = [
        ROOT / "motrix_envs/src/motrix_envs/locomotion/wbt/g1/ground_action_ablation.py",
        ROOT / "motrix_envs/src/motrix_envs/locomotion/wbt/g1/__init__.py",
        Path(__file__).resolve(),
    ]
    sources.extend(ROOT / "configs/task" / (PREFIX + suffix) / "motrix.fastsac.yaml" for suffix, _, _ in GROUPS)
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(out / "source-sha256.json", hashes)
    (out / "tracked-source.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT))
    rows = []
    for suffix, weight, friction in GROUPS:
        name = PREFIX + suffix
        group_dir = out / suffix
        group_dir.mkdir()
        command = [python, mxlab, "train", f"task={name}/motrix.fastsac"]
        with (group_dir / "resolved-task.yaml").open("w") as log:
            subprocess.run(command + ["--cfg", "job"], cwd=ROOT, stdout=log, check=True)
        wait_for_gpu(gpu)
        parent = ROOT / "runs" / name / "motrix/torch/fastsac"
        before = set(parent.glob("*")) if parent.exists() else set()
        row = dict(
            group=suffix,
            action_rate_weight=weight,
            friction=friction,
            iterations=100000,
            commit=commit,
            start=now(),
            status="running",
        )
        write_json(group_dir / "status.json", row)
        print(now(), "Starting", suffix, flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
        with (group_dir / "train.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        new_runs = set(parent.glob("*")) - before
        row.update(
            end=now(),
            exit_code=result.returncode,
            status="completed" if result.returncode == 0 else "failed",
            run_dir=str(next(iter(new_runs))) if len(new_runs) == 1 else None,
        )
        write_json(group_dir / "status.json", row)
        rows.append(row)
        with (out / "summary.tsv").open("w") as summary:
            writer = csv.DictWriter(summary, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(now(), row["status"], suffix, row["run_dir"], flush=True)
    # TensorBoard may import TensorFlow. Hide GPUs before importing it after training.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    for row in rows:
        metrics = {}
        if row["run_dir"]:
            accumulator = EventAccumulator(row["run_dir"], size_guidance={"scalars": 0})
            accumulator.Reload()
            tags = set(accumulator.Tags()["scalars"])
            for tag in (
                "metrics/box_lift_verified",
                "metrics/box_success_total",
                "metrics/box_success",
                "metrics/object_position_error",
                "metrics/object_orientation_error",
                "metrics/box_curriculum_survival_ratio",
                "metrics/box_reward_weight",
                "metrics/object_termination_enabled",
            ):
                if tag in tags:
                    values = accumulator.Scalars(tag)
                    if values:
                        metrics[tag] = dict(
                            final_step=values[-1].step,
                            final=values[-1].value,
                            maximum=max(value.value for value in values),
                        )
        write_json(out / row["group"] / "metrics.json", metrics)
        row["metrics"] = metrics
    write_json(out / "results.json", dict(end=now(), experiments=rows))
    print(now(), "Queue finished", out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This queue is already running") from None
        if (out / "queue.json").exists():
            raise SystemExit("This queue was already started; choose a new output directory")
        run_queue(out, args.gpu)


if __name__ == "__main__":
    main()
