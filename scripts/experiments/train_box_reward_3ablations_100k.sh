#!/usr/bin/env bash
# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0
#
# Sequential 100k-iteration FastSAC reward ablations.
# Run under tmux after confirming the GPU is idle and required motion NPZ exists.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MOTION="motrix_envs/src/motrix_envs/locomotion/wbt/assets/motion/g1/sub3_largebox_003_upright_start_end_hold_745.npz"
if [[ ! -f "$MOTION" ]]; then
  echo "ERROR: missing 745-frame reference NPZ: $MOTION" >&2
  exit 1
fi

mkdir -p runs
# Prevent two copies of the overnight ablation runner from sharing one GPU.
exec 9>runs/.box_reward_3ablations_100k.lock
if ! flock -n 9; then
  echo "ERROR: another three-ablation runner already holds the lock" >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/box_reward_3ablations_100k/$RUN_ID"
mkdir -p "$LOG_DIR"
{
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_branch=$(git branch --show-current)"
  echo "start=$(date -Is)"
  echo "seed=1"
  echo "num_envs=2048"
  echo "iterations_per_experiment=100000"
  echo "motion=$MOTION"
  git status --short
} >"$LOG_DIR/metadata.txt"

TASK="g1-29dof-wbt-largebox-omniretarget-fastsac/motrix.fastsac"
PREFIX="g1-29dof-wbt-largebox-omniretarget-fastsac"

run_experiment() {
  local label="$1"
  local env_suffix="$2"
  local env_name="${PREFIX}-${env_suffix}"
  echo "[$(date -Is)] START $label: $env_name / 100000 iterations"
  # A failed trainer aborts this script; later experiments are not launched.
  HYDRA_FULL_ERROR=1 uv run mxlab train \
    "task=$TASK" \
    "task.env=$env_name" \
    num_envs=2048 \
    seed=1 \
    algo.trainer.num_learning_iterations=100000 \
    play=false \
    2>&1 | tee "$LOG_DIR/${label}.log"
  date -Is >"$LOG_DIR/${label}.completed"
  echo "[$(date -Is)] COMPLETE $label"
}

trap 'echo "[$(date -Is)] ERROR: overnight ablation runner stopped; see logs in $LOG_DIR" >&2' ERR
run_experiment "01_reward05_80" "reward05-80"
run_experiment "02_reward05_50" "reward05-50"
run_experiment "03_reward05_50_orifixed" "reward05-50-orifixed"
date -Is >"$LOG_DIR/ALL_COMPLETED"
echo "[$(date -Is)] ALL THREE ABLATIONS COMPLETED: $LOG_DIR"
