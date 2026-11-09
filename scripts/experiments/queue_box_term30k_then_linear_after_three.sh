#!/usr/bin/env bash
# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0
#
# Install OUTSIDE the active worktree with 'git show <remote>/<branch>:<path>'.
# Wait for all three original experiments to finish; then switch branches and
# run the four subsequent 100k experiments (fixed term30k, then 3 linear).
set -Eeuo pipefail

: "${WORKTREE_ROOT:?Set WORKTREE_ROOT to the existing MotrixLab worktree}"
: "${PRIOR_RUN_DIR:?Set PRIOR_RUN_DIR to the exact old 3-ablation log directory}"
: "${FORK_REMOTE:?Set FORK_REMOTE to the existing remote for 2522514190/MotrixLab}"
PRIOR_TMUX_SESSION="${PRIOR_TMUX_SESSION:-g1_box_reward_3x100k}"
TARGET_BRANCH="codex/fastsac-term30k-then-linear-4x100k"
cd "$WORKTREE_ROOT"
PRIOR_RUN_DIR="$(realpath "$PRIOR_RUN_DIR")"

# Never allow both the previously queued linear-only watcher and this new
# watcher to race for the same worktree. Codex must inspect and safely disable
# the OLD *waiting queue* first, without touching the actual old training.
if tmux has-session -t g1_box_reward_linear_queue 2>/dev/null; then
  echo "ERROR: old linear-only queue session is still active." >&2
  echo "Inspect/disable it safely before starting the new four-stage queue." >&2
  exit 1
fi

if [[ ! -d "$PRIOR_RUN_DIR" ]]; then
  echo "ERROR: original experiments' log directory missing: $PRIOR_RUN_DIR" >&2
  exit 1
fi

echo "[$(date -Is)] Waiting for original experiments 1-3 in $PRIOR_RUN_DIR"
while [[ ! -f "$PRIOR_RUN_DIR/ALL_COMPLETED" ]]; do
  if ! tmux has-session -t "$PRIOR_TMUX_SESSION" 2>/dev/null; then
    echo "ERROR: original tmux stopped without ALL_COMPLETED; not starting next run." >&2
    exit 1
  fi
  sleep 30
done

for marker in 01_reward05_80.completed 02_reward05_50.completed 03_reward05_50_orifixed.completed; do
  if [[ ! -f "$PRIOR_RUN_DIR/$marker" ]]; then
    echo "ERROR: original experiment $marker not complete" >&2
    exit 1
  fi
done

# The original session may still be writing its final log. Never switch until
# its tmux session exits naturally after the completion markers are written.
while tmux has-session -t "$PRIOR_TMUX_SESSION" 2>/dev/null; do
  sleep 5
done

if pgrep -af 'mxlab train|motrix_rl.commands.train' > /dev/null; then
  echo "ERROR: another MotrixLab trainer is active; refusing to switch branch." >&2
  pgrep -af 'mxlab train|motrix_rl.commands.train' >&2 || true
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: tracked local modifications found; refusing to switch branch." >&2
  git status --short >&2
  exit 1
fi
if tmux has-session -t g1_box_reward_linear_queue 2>/dev/null; then
  echo "ERROR: old linear-only queue appeared; refusing to switch." >&2
  exit 1
fi

echo "[$(date -Is)] Original three experiments finished. Fetching new branch."
git fetch "$FORK_REMOTE" "+refs/heads/$TARGET_BRANCH:refs/remotes/$FORK_REMOTE/$TARGET_BRANCH"
EXPECTED_SHA="$(git rev-parse "refs/remotes/$FORK_REMOTE/$TARGET_BRANCH")"
if git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH"; then
  git switch "$TARGET_BRANCH"
else
  git switch --track -c "$TARGET_BRANCH" "$FORK_REMOTE/$TARGET_BRANCH"
fi
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_SHA" ]]; then
  echo "ERROR: local HEAD != fetched remote HEAD; refusing to run." >&2
  exit 1
fi

echo "[$(date -Is)] Running tests before four consecutive 100k jobs."
bash -n scripts/experiments/train_box_term30k_then_linear_4x100k.sh
uv run python -m pytest -q tests/test_omniretarget_fastsac.py \
  2>&1 | tee "$PRIOR_RUN_DIR/term30k_then_linear_followon_tests.log"

echo "[$(date -Is)] Test suite passed. Starting fixed-30k termination ablation."
PRIOR_RUN_DIR="$PRIOR_RUN_DIR" bash scripts/experiments/train_box_term30k_then_linear_4x100k.sh
