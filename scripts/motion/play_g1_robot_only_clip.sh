#!/usr/bin/env bash
# Reuse the box simulation runtime, removing only the physical box.
set -euo pipefail
: "${1:?Usage: bash scripts/motion/play_g1_robot_only_clip.sh ARTIFACT [Hydra overrides...]}"
artifact="$(realpath "$1")"
shift
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
exec "$project_dir/.venv/bin/python" "$project_dir/.venv/bin/mxlab" deploy \
 task=g1-largebox/sim \
 "artifact=$artifact" \
 runtime.box_enabled=false \
 runtime.physics.dt=0.002 \
 runtime.physics.solver_iterations=100 \
 runtime.push.enabled=false \
 runtime.randomization.enabled=true \
 "$@"
