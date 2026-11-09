#!/usr/bin/env bash
# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0
#
# This script is intended to be copied OUTSIDE the active worktree via
# 'git show <remote-branch>:scripts/experiments/queue_box_reward_linear_after_three.sh'
# and then run in its own tmux session while the original 3-ablation job runs.
# It never switches branches until experiment 3 and the original tmux session finish.
set -Eeuo pipefail

: "${WORKTREE_ROOT:?Set WORKTREE_ROOT to the existing MotrixLab worktree}"
: "${PRIOR_RUN_DIR:?Set PRIOR_RUN_DIR to the exact prior 3-ablation log directory}"
: "${FORK_REMOTE:?Set FORK_REMOTE to the existing remote for 2522514190/MotrixLab}"
PRIOR_TMUX_SESSION="${PRIOR_TMUX_SESSION:-g1_box_reward_3x100k}"

TARGET_BRANCH="codex/fastsac-box-reward-linear-iters-3x100k"
cd "$WORKTREE_ROOT"

if [[ ! -d "$PRIOR_RUN_DIR" ]]; then
  echo "ERROR: prior experiment run directory not found: $PRIOR_RUN_DIR" >&2
  exit 1
fi
echo "[$(date -Is)] Waiting for experiment three to complete: $PRIOR_RUN_DIR"

while [[ ! -f "$PRIOR_RUN_DIR/ALL_COMPLETED" ]]; do
  if ! tmux has-session -t "$PRIOR_TMUX_SESSION" 2>/dev/null; then
    echo "ERROR: prior training session ended without ALL_COMPLETED; not starting follow-on." >&2
    exit 1
  fi
  sleep 30
done

for marker in 01_reward05_80.completed 02_reward05_50.completed 03_reward05_50_orifixed.completed; do
  if [[ ! -f "$PRIOR_RUN_DIR/$marker" ]]; then
    echo "ERROR: prior run has ALL_COMPLETED but is missing $marker" >&2
    exit 1
  fi
done

# The completion marker is written just before the old script exits. The old
# tmux session must exit before the same worktree can be safely switched.
while tmux has-session -t "$PRIOR_TMUX_SESSION" 2>/dev/null; do
  sleep 5
done

# Do not steal the GPU or switch files out from under an unrelated run.
if pgrep -af 'mxlab train|motrix_rl.commands.train' > /dev/null; then
  echo "ERROR: another training process is active; refusing to switch branch." >&2
  pgrep -af 'mxlab train|motrix_rl.commands.train' >&2 || true
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: tracked worktree files changed locally; manual reconciliation required." >&2
  git status --short >&2
  exit 1
fi

echo "[$(date -Is)] Prior three experiments completed. Fetching follow-on branch."
git fetch "$FORK_REMOTE" "+refs/heads/$TARGET_BRANCH:refs/remotes/$FORK_REMOTE/$TARGET_BRANCH"
EXPECTED_SHA="$(git rev-parse "refs/remotes/$FORK_REMOTE/$TARGET_BRANCH")"
if git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH"; then
  git switch "$TARGET_BRANCH"
else
  git switch --track -c "$TARGET_BRANCH" "$FORK_REMOTE/$TARGET_BRANCH"
fi
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_SHA" ]]; then
  echo "ERROR: checked-out branch does not match fetched remote commit." >&2
  exit 1
fi

echo "[$(date -Is)] Validating follow-on experiment code at commit $EXPECTED_SHA"
python -m py_compile motrix_envs/src/motrix_envs/locomotion/wbt/g1/omniretarget_fastsac.py
bash -n scripts/experiments/train_box_reward_linear_3ablations_100k.sh
uv run python -m pytest -q tests/test_omniretarget_fastsac.py \
  2>&1 | tee "$PRIOR_RUN_DIR/linear_followon_tests.log"
echo "[$(date -Is)] Tests passed. Starting linear iteration reward experiments."
PRIOR_RUN_DIR="$PRIOR_RUN_DIR" bash scripts/experiments/train_box_reward_linear_3ablations_100k.sh
