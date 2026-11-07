#!/usr/bin/env bash
set -u -o pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
STAMP=$(date +%Y%m%d-%H%M%S)
OUT_ROOT="${FASTSAC_ABLATION_ROOT:-$ROOT/runs/g1-largebox-fastsac-ablation-$STAMP}"
mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.tsv"
TOTAL_LOG="$OUT_ROOT/overnight.log"
COMMIT=$(git rev-parse HEAD 2>/dev/null || printf unknown)
printf 'group\tcontact_penalty\tbox_error_termination\tadaptive_sampling\titerations\tstatus\tstart\tend\trun_dir\n' > "$SUMMARY"
printf '[%s] commit=%s gpu=%s curriculum=full-start-survival-ema/reference_frames=325\n' "$(date -Is)" "$COMMIT" "$CUDA_VISIBLE_DEVICES" | tee -a "$TOTAL_LOG"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee "$OUT_ROOT/gpu-start.txt" >> "$TOTAL_LOG"; fi

groups=(A B C D E F G H)
contacts=(-0.1 0 -0.1 -0.1 0 0 -0.1 0)
box_terms=(true true false true false true false false)
adaptive=(true true true false true false false false)
find_latest_run() { find "$ROOT/runs" -type f -name metadata.json -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-; }
for i in "${!groups[@]}"; do
  group=${groups[$i]}; env_name="g1-29dof-wbt-largebox-omniretarget-fastsac-${group,,}"; group_dir="$OUT_ROOT/$group"; mkdir -p "$group_dir"
  start=$(date -Is); before=$(find_latest_run || true)
  printf '[%s] START %s env=%s contact=%s box_error_termination=%s adaptive_sampling=%s\n' "$start" "$group" "$env_name" "${contacts[$i]}" "${box_terms[$i]}" "${adaptive[$i]}" | tee -a "$TOTAL_LOG" "$group_dir/run.log"
  printf 'commit=%s\nseed=1\nnum_learning_iterations=40000\ncontact_penalty=%s\nbox_error_termination=%s\nadaptive_sampling=%s\ncurriculum=full-start-survival-ema/reference_frames=325\n' "$COMMIT" "${contacts[$i]}" "${box_terms[$i]}" "${adaptive[$i]}" > "$group_dir/experiment.txt"
  mxlab train task=g1-29dof-wbt-largebox-omniretarget-fastsac/motrix.fastsac task.env="$env_name" seed=1 algo.trainer.num_learning_iterations=40000 2>&1 | tee -a "$group_dir/train.log" "$TOTAL_LOG"
  rc=${PIPESTATUS[0]}
  end=$(date -Is); after=$(find_latest_run || true); run_dir="$after"
  if [[ "$after" == "$before" ]]; then run_dir=""; fi
  status=failed
  if [[ $rc -eq 0 ]]; then status=completed; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$group" "${contacts[$i]}" "${box_terms[$i]}" "${adaptive[$i]}" 40000 "$status" "$start" "$end" "$run_dir" >> "$SUMMARY"
  printf '[%s] END %s status=%s rc=%s run_dir=%s\n' "$end" "$group" "$status" "$rc" "$run_dir" | tee -a "$TOTAL_LOG" "$group_dir/run.log"
  printf 'start=%s\nend=%s\nexit_status=%s\nrun_dir=%s\n' "$start" "$end" "$rc" "$run_dir" >> "$group_dir/experiment.txt"
done
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader | tee "$OUT_ROOT/gpu-end.txt" >> "$TOTAL_LOG"; fi
printf '[%s] ALL GROUPS FINISHED; summary=%s\n' "$(date -Is)" "$SUMMARY" | tee -a "$TOTAL_LOG"
