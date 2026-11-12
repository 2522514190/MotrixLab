#!/usr/bin/env bash
set -euo pipefail
: "${1:?Usage: bash run_robot_only_clip.sh NETWORK_INTERFACE}"
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python -u -c 'from motrix_deploy.cli import main; main()' \
 task=g1-box-pc2/hardware \
 "artifact=$bundle_dir/artifacts/g1-robot-only-clip-pc2.deploy" \
 "runtime.transport.network_interface=$1"
