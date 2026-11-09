#!/usr/bin/env bash
# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0
#
# Follow-on experiment: run three FastSAC 100k-iteration linear reward decays,
# strictly after the prior three survival-scheduled ablations have finished.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EXPECTED_BRANCH="codex/fastsac-box-reward-linear-iters-3x100k"
ACTUAL_BRANCH="$(git branch --show-current)"
if [[ "$ACTUAL_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "ERROR: expected Git branch $EXPECTED_BRANCH, got $ACTUAL_BRANCH" >&2
  exit 1
fi

# Select the specific prior run if the caller knows it, to avoid accidentally
# accepting an old completion marker from another night.
PRIOR_RUN_DIR="${PRIOR_RUN_DIR:-}"
if [[ -z "$PRIOR_RUN_DIR" ]]; then
  # Previous script stores one timestamped directory per 3-experiment sequence.
  PRIOR_ROOT="runs/box_reward_3ablations_100k"
  if [[ ! -d "$PRIOR_ROOT" ]]; then
    echo "ERROR: previous experiment log directory missing: $PRIOR_ROOT" >&2
    exit 1
  fi
  PRIOR_LAST="$(find "$PRIOR_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail -n 1)"
  if [[ -z "$PRIOR_LAST" ]]; then
    echo "ERROR: no previous experiment run directories found" >&2
    exit 1
  fi
  PRIOR_RUN_DIR="$PRIOR_ROOT/$PRIOR_LAST"
fi

for marker in "01_reward05_80.completed" "02_reward05_50.completed" "03_reward05_50_orifixed.completed" "ALL_COMPLETED"; do
  if [[ ! -f "$PRIOR_RUN_DIR/$marker" ]]; then
    echo "ERROR: prior run is not complete ($PRIOR_RUN_DIR/$marker absent)." >&2
    echo "Do not start this follow-on run until experiment 3 finishes successfully." >&2
    exit 1
  fi
done

MOTION="motrix_envs/src/motrix_envs/locomotion/wbt/assets/motion/g1/sub3_largebox_003_upright_start_end_hold_745.npz"
if [[ ! -f "$MOTION" ]]; then
  echo "ERROR: missing reference NPZ: $MOTION" >&2
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
print(f"Reference OK: {frames} frames, {fps} Hz")
PY

mkdir -p runs
exec 9>runs/.box_reward_linear_3ablations_100k.lock
if ! flock -n 9; then
  echo "ERROR: another linear reward three-ablation launcher is running" >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/box_reward_linear_3ablations_100k/$RUN_ID"
mkdir -p "$LOG_DIR"
{
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "start=$(date -Is)"
  echo "prior_complete_run=$PRIOR_RUN_DIR"
  echo "seed=1"
  echo "num_envs=2048"
  echo "iterations_per_run=100000"
  echo "box_reward_schedule=linear_by_collector_iterations"
  git status --short
} > "$LOG_DIR/metadata.txt"

TASK="g1-29dof-wbt-largebox-omniretarget-fastsac/motrix.fastsac"
PREFIX="g1-29dof-wbt-largebox-omniretarget-fastsac"
run_experiment() {
  local label="$1"
  local env_suffix="$2"
  local env_name="${PREFIX}-${env_suffix}"
  echo "[$(date -Is)] START $label: $env_name"
  HYDRA_FULL_ERROR=1 uv run mxlab train \
    "task=$TASK" \
    "task.env=$env_name" \
    num_envs=2048 \
    seed=1 \
    algo.trainer.num_learning_iterations=100000 \
    play=false \
    2>&1 | tee "$LOG_DIR/${label}.log"
  date -Is > "$LOG_DIR/${label}.completed"
  echo "[$(date -Is)] COMPLETED $label"
}
trap 'echo "[$(date -Is)] ERROR: linear reward training stopped. Logs: $LOG_DIR" >&2' ERR

run_experiment "04_linear08_100k" "linear08-100k"
run_experiment "05_linear05_100k" "linear05-100k"
run_experiment "06_linear10_100k" "linear10-100k"
date -Is > "$LOG_DIR/ALL_COMPLETED"
echo "[$(date -Is)] ALL LINEAR DECAY EXPERIMENTS FINISHED: $LOG_DIR"
