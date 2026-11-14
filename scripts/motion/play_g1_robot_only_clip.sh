#!/usr/bin/env bash
# Deploy the robot-only G1 on the flat scene.
set -euo pipefail
: "${1:?Usage: bash scripts/motion/play_g1_robot_only_clip.sh ARTIFACT [Hydra overrides...]}"
artifact="$(realpath "$1")"
shift
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
exec "$project_dir/.venv/bin/python" "$project_dir/.venv/bin/mxlab" deploy \
 task=g1-robot-only/sim \
 "artifact=$artifact" \
 runtime.box_enabled=false \
 runtime.physics.dt=0.002 \
 runtime.physics.solver_iterations=100 \
 runtime.push.enabled=false \
 runtime.randomization.enabled=true \
 "$@"
