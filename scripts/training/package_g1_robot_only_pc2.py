"""Package a newly exported robot-only policy for the existing PC2 box runtime."""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output / "artifacts/g1-robot-only-clip-pc2.deploy"
    shutil.copytree(args.artifact, target)
    manifest_file = target / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["task"]["name"] = "g1_wbt_box_pc2/v1"
    assert manifest["task"]["config"]["action_safety_clip_fraction"] == 0.95
    assert manifest["task"]["config"]["motion_frames"] == 745
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    launcher = args.output / "run_robot_only_clip.sh"
    launcher.write_text("""#!/usr/bin/env bash
set -euo pipefail
: "${1:?Usage: bash run_robot_only_clip.sh NETWORK_INTERFACE}"
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python -u -c 'from motrix_deploy.cli import main; main()' \\
 task=g1-box-pc2/hardware \\
 "artifact=$bundle_dir/artifacts/g1-robot-only-clip-pc2.deploy" \\
 "runtime.transport.network_interface=$1"
""")
    launcher.chmod(0o755)
    (args.output / "README.md").write_text(
        "# 无箱子 clip 策略（PC2）\n\n"
        "在已安装并验证过的 PC2 双 WBT v2 环境运行 `bash run_robot_only_clip.sh 网卡名`。\n"
        "复用 g1-box-pc2/hardware 和现有 wbt_box Python 代码；无需升级依赖。\n"
        "新模型从头训练；无箱子轨迹 745 帧 / 50 Hz；目标角度 clip=0.95。\n"
        "沿用原启动按键与航向对齐，14.9 秒动作后保持末帧，总计 600 秒，再阻尼。\n"
        "仿真评估和 ONNX 数值报告见训练输出目录。真实机器人运行尚未验证。\n"
    )
    print(target.resolve())


if __name__ == "__main__":
    main()
