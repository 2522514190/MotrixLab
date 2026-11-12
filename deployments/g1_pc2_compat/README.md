# PC2 双 WBT 离线部署包 v2

仅适用于已通过 SHA256 确认的 PC2 9e499a6 + final-hold-600 兼容基线。不是之前的新版 robot-only 插件包。

## 原则

不删除或覆盖 PC2 已验证的公共部署文件，不升级 SDK、native 扩展或依赖。安装脚本校验基线、保存三个关键源码文件、离线安装一个独立插件、再次校验公共文件没有变化，再运行禁止网络与硬件导入的验收。PC2 当前旧补丁已经精确识别，没有理由执行 git reset 或清空旧目录。旧命令留作回退，新任务执行已修复航向对齐的独立模块。

## 安装

先退出正在运行的控制程序。激活 PC2 原环境后，在本包目录执行：

```bash
bash install_offline.sh
```

不用联网；要求现有环境能运行旧 CLI，并已有 pip 或 uv。安装阶段不运行任何机器人控制。

## 分别启动（仅安装验收通过、现场准备好后）

```bash
bash run_box.sh eth0
bash run_robot_only.sh eth0
```

以上两个命令只能选择一个运行，不能同时控制同一台 G1。都使用旧版 Python CLI，不依赖 mxlab。

带箱子：task=g1-box-pc2/hardware，类型 g1_wbt_box_pc2/v1，模块 motrix_deploy_pc2.wbt_box。保留原模型 78000、clip=0.95；14.9 秒播放 + 585.1 秒末帧保持，然后阻尼。可通过手动 CLI 的 task_settings.final_frame_hold_s 缩短保持时间。

无箱子：task=g1-robot-only-pc2/hardware，类型 g1_wbt_robot_only_pc2/v1，模块 motrix_deploy_pc2.wbt_robot_only。保留原 robot-only latest 模型，无目标角度 clip 或开关；14.9 秒播放后持续保持最后一帧，直到手动停止或保护触发。

两者都在策略接管时对齐参考航向；保留旧 Start/渐变/新 A 按键边沿、Preparation、急停和硬件 watchdog。网卡参数仍为 runtime.transport.network_interface，不使用新版 runtime.network_interface。

## 旧模型审计

旧 manifest 声明 xyzw、body_to_world、body-frame gyro；154 输入、29 输出；745 帧/50 Hz；源 checkpoint=model_0078000.pt。模型及轨迹 payload SHA 与原文件一致。首帧 torso yaw=-1.607507 rad。旧兼容包只定义 align_reference_to，未调用。新模块补上接管时调用，不改模型权重或轨迹。

## 验证与范围

在本地物化的旧 9e499a6 + 与 PC2 相同的两个兼容文件下：旧工厂真实加载两个独立类型、两份实际 ONNX 初始化及推理、带箱子 0.95 裁剪/无箱子非裁剪、三种初始航向、冻结末帧持续推理和停止检查通过。46 个旧启动/WBT/600 秒逻辑时钟测试通过；旧版 Hydra 硬件配置合成通过。没有导入硬件模块或创建网络连接。

这不是对真机稳定性的保证，也没有证明历史 sim2real 失败只有一个原因。首次真机验证使用吊架和操作员急停，先验证启动接管，再验证整段；不要把首次验证直接当作 10 分钟稳定性测试。

## 回退

原公共代码、旧 artifact 和旧命令未更改；退出新程序后可以回到原命令。若需要删除新增插件，在原环境用 pip uninstall motrix-deploy-pc2-wbt 或 uv pip uninstall --python 对应Python motrix-deploy-pc2-wbt。安装前文件备份位于 ~/g1-pc2-before-v2-*。
