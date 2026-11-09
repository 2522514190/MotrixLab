#!/usr/bin/env bash
# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0
#
# A single 30k iteration-activated box termination ablation, followed by three
# independent linear box reward ablations. Four fresh 100k FastSAC runs in order.
# To run after existing jobs, use queue_box_term30k_then_linear_after_three.sh.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BRANCH="codex/fastsac-term30k-then-linear-4x100k"
if [[ "$(git branch --show-current)" != "$BRANCH" ]]; then
  echo "ERROR: expected Git branch $BRANCH" >&2
  exit 1
fi

: "${PRIOR_RUN_DIR:?Set PRIOR_RUN_DIR to the exact directory for completed original experiments 1-3}"
PRIOR_RUN_DIR="$(realpath "$PRIOR_RUN_DIR")"
for marker in 01_reward05_80.completed 02_reward05_50.completed 03_reward05_50_orifixed.completed ALL_COMPLETED; do
  if [[ ! -f "$PRIOR_RUN_DIR/$marker" ]]; then
    echo "ERROR: original experiments not all finished ($PRIOR_RUN_DIR/$marker missing)" >&2
    exit 1
  fi
done

MOTION="motrix_envs/src/motrix_envs/locomotion/wbt/assets/motion/g1/sub3_largebox_003_upright_start_end_hold_745.npz"
if [[ ! -f "$MOTION" ]]; then
  echo "ERROR: missing 745-frame reference NPZ: $MOTION" >&2
  exit 1
fi
uv run python - "$MOTION" <<'PY'
import sys
import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as clip:
    frames = int(clip["num_frames"])
    fps = int(clip["fps"])
    assert (frames, fps) == (745, 50), (frames, fps)
    assert np.isfinite(clip["joint_pos"]).all()
print(f"Reference OK: {frames} frames at {fps} Hz")
PY

mkdir -p runs
exec 9>runs/.box_term30k_then_linear_4x100k.lock
if ! flock -n 9; then
  echo "ERROR: another four-ablation launcher is running" >&2
  exit 1
fi
# Hold the original linear-runner lock too, preventing the old queued
# linear-only runner from launching concurrently with this four-stage run.
exec 8>runs/.box_reward_linear_3ablations_100k.lock
if ! flock -n 8; then
  echo "ERROR: linear-only runner is active" >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/box_term30k_then_linear_4x100k/$RUN_ID"
mkdir -p "$LOG_DIR"
{
  echo "start=$(date -Is)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "prior_run_dir=$PRIOR_RUN_DIR"
  echo "seed=1"
  echo "num_envs=2048"
  echo "iterations_per_experiment=100000"
  echo "experiment_04=reward05-80-term30k;object_term_pos=1.0m;object_term_ori=0.8rad;gate_at=30000"
  echo "experiments_05_06_07=linear08-100k,linear05-100k,linear10-100k"
  git status --short
} >"$LOG_DIR/metadata.txt"

TASK="g1-29dof-wbt-largebox-omniretarget-fastsac/motrix.fastsac"
PREFIX="g1-29dof-wbt-largebox-omniretarget-fastsac"
run_experiment() {
  local label="$1" suffix="$2" env_name="${PREFIX}-${suffix}"
  echo "[$(date -Is)] START $label ($env_name) 100000 iterations"
  HYDRA_FULL_ERROR=1 uv run mxlab train \
    "task=$TASK" \
    "task.env=$env_name" \
    num_envs=2048 \
    seed=1 \
    algo.trainer.num_learning_iterations=100000 \
    play=false 2>&1 | tee "$LOG_DIR/${label}.log"
  date -Is >"$LOG_DIR/${label}.completed"
  echo "[$(date -Is)] COMPLETE $label"
}
trap 'echo "[$(date -Is)] ERROR: stopped; inspect $LOG_DIR" >&2' ERR

run_experiment "04_reward05_80_term30k" "reward05-80-term30k"
run_experiment "05_linear08_100k" "linear08-100k"
run_experiment "06_linear05_100k" "linear05-100k"
run_experiment "07_linear10_100k" "linear10-100k"
date -Is >"$LOG_DIR/ALL_COMPLETED"
echo "[$(date -Is)] ALL FOUR EXPERIMENTS FINISHED: $LOG_DIR"
