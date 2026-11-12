# G1 无箱子 WBT 独立部署包

本包新增 motrix-deploy-robot-only 插件，不替换现有 motrix_deploy_tasks、dance、带箱子代码，不安装或升级依赖。不使用此前要求覆盖 g1_wbt.py 的迁移方式。

在 G1 控制电脑激活原来能运行 dance/带箱子的环境，然后在本包目录运行：

```bash
bash install_offline.sh
bash run_robot_only.sh 实际网卡名
```

安装一次即可。此后旧策略仍使用原来的命令，新策略使用 run_robot_only.sh。无需切换环境。

新入口：task=g1-wbt-robot-only/hardware。
新 artifact：artifacts/g1-largebox-robot-only.deploy。
新 manifest 类型：g1_wbt_robot_only/v1。
新执行模块：motrix_deploy_robot_only.g1_wbt。
原来的 g1_wbt/v1 仍由原来的 motrix_deploy_tasks.tasks.g1_wbt 处理。

无 clip 开关，无目标角度裁剪。轨迹 745 帧/50 Hz，自动播放 14.9 秒后保持末帧；航向在策略接管时对齐；手动停止或保护触发时结束。无需设置 dance 的 duration_s。

如果此前已经覆盖过 G1 上的旧 g1_wbt.py，需要先从自己的备份恢复旧执行代码；安装这个插件不会自动还原此前覆盖的文件。

验证：新类型成功注册，与原类型分别路由；新 wheel 不包含原包文件；ONNX 32 样本校验通过（最大绝对误差 8.34465e-06）；独立执行模块 32 项生命周期/动作/航向回归检查通过；新类型 MotrixSim 完整播放及末帧保持 5 秒通过。未连接实机。首次实机运行使用吊架并保留操作员急停。

同一时间只运行一个控制 G1 的程序。安装要求现有 motrix-deploy、motrix-env-core、motrix-deploy-tasks 版本为 0.4.0；不同版本请先核对兼容性，安装脚本不会升级它们。
