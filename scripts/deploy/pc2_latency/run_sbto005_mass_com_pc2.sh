#!/usr/bin/env bash
set -euo pipefail
cd /home/unitree/MotrixLab
source .venv/bin/activate
mkdir -p logs/wbt
log_file="$(mktemp "logs/wbt/sbto005-mass-com-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")"
export PC2_TIMING_PREFIX="/home/unitree/MotrixLab/${log_file%.log}"
printf '运行日志：%s\n延迟统计：%s.timing.json\n' "$log_file" "$PC2_TIMING_PREFIX"
# Stage messages remain visible; per-cycle timings are buffered in memory.
python -u /home/unitree/g1-sbto005-latency/run_latency.py \
  task=g1-wbt/hardware \
  artifact=/home/unitree/MotrixLab/artifacts/sbto005-mass-com.deploy \
  "runtime.transport.network_interface=${1:-eth0}" \
  +task_settings.frame0_start=true \
  task_settings.preparation.default_transition_duration_s=2.0 \
  +task_settings.final_frame_hold_s=0 "${@:2}" 2>&1 | tee -a "$log_file"
