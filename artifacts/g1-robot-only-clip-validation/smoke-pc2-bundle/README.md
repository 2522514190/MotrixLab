# 无箱子 clip 策略（PC2）

在已安装并验证过的 PC2 双 WBT v2 环境运行 `bash run_robot_only_clip.sh 网卡名`。
复用 g1-box-pc2/hardware 和现有 wbt_box Python 代码；无需升级依赖。
新模型从头训练；无箱子轨迹 745 帧 / 50 Hz；目标角度 clip=0.95。
沿用原启动按键与航向对齐，14.9 秒动作后保持末帧，总计 600 秒，再阻尼。
仿真评估和 ONNX 数值报告见训练输出目录。真实机器人运行尚未验证。
