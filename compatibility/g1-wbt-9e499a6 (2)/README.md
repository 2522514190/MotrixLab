# G1 WBT / PC2 9e499a6 兼容回移

## 审计基线

- 本地 `codex/box-carry-training` = `1ddb974`，三个 deploy 包与 `9e499a6` 无差异。
- 当前目录实际 HEAD = `f6abf9e`（`codex/fastsac-no-target-clip-bimanual-contact`），包含用户描述的新接口。没有切换分支或覆盖当前分支源码。
- 本包专用于 PC2 的 `9e499a6`，不能覆盖到已升级整套新版硬件接口的环境。

## 调用与兼容性结论

旧版：`mxlab deploy → motrix_deploy.cli.run → read_artifact → OnnxPolicy → create_task(settings=cfg.task_settings) → G1WbtDeployTask(settings=...) → configure_operator → prepare → ControlSession → HardwareRuntime`。

旧版 WBT 将 `settings.preparation` 严格解析为 `G1StartupSettings`；未知设置报错。硬件 `prepare` 强制要求 GamePadDevice。Start 确认后从实测姿态向默认站姿插值，等待新的 A 按键边沿；提前按住 A 不会授权。A 放行后才进入任务自身的 Preparation。Preparation 仍需渐变、连续满足实测姿态/速度要求、倾角检查和超时检查，再进入 policy_hold，至少一个冻结参考帧控制周期后 playback，最后 damping。`automatic_start=False` 仍要求显式 request_policy_start/request_motion_start；默认 True 只控制上述 Preparation 之后的策略接管，不替代 Start/A。

新版工厂和 CLI 均已删除 settings/task_settings；WBT 删除 startup、configure_operator/prepare 接线，默认 automatic_start=True。直接换新版 task.py 会掩盖启动链变化；保留旧工厂只换新版 WBT 则出现当前 TypeError。新版仅 attach_keyboard 后才处理 p/m/o；继承的 prepare 不会自动接上手柄确认或键盘。

其他跨版本变化：

- create_hardware_runtime 从 `(config, context)` 变为 `(name, config, context)`。
- Unitree config 从包内 transport/control 模式改为单文件及 robot 配置；hardware.yaml、plugin、native/python adapter 均迁移。
- ControlSession 从 read_state() 改为 read_state(timeout)，HardwareRuntime 不再在原位置 acquire_control。
- JointServoCommand 的 copy/validate_for 被删除；与旧硬件链混用有后续风险。
- 新仿真初态需要 CLI configure_simulation 和新版 simulator 配合；本回移只解析/保留该 Artifact 元数据，不移植仿真直接 playback，也不让它绕过真机 Preparation。
- 旧 dance/hardware preset 固定 duration_s=19.98，但当前 Artifact 745 帧 × 0.02 = 14.9 秒，CLI 会拒绝超预算；使用 duration_s=null 取 Artifact 默认，或明确 14.9。

## 最小文件集合

1. `g1_wbt.py` → `motrix_deploy_tasks/src/motrix_deploy_tasks/tasks/g1_wbt.py`：以完整旧版为基础，只加入新 Artifact 两个字段及动作裁剪实现。
2. `config.py` → `motrix_deploy/src/motrix_deploy/runtime/config.py`：新增初态数据类，是上面字段解析的依赖；内容与用户已经同步的新版相同。

PC2 已有正确 config.py 时，本次实际只需替换 g1_wbt.py。保留 PC2 旧版 g1_startup.py、task.py、cli.py、contracts.py、runtime 其他文件、Unitree 整包及旧 hardware.yaml。无需新依赖、重编 native 或修改 Artifact。Artifact 是目录，传输须保留 policy、payloads、manifest。

安全行为：旧版手柄确认、operator intents、后端 select 急停/被动请求、命令 watchdog、状态超时及停机 damping 均保留。策略位置目标按关节物理区间中点裁到 Artifact 指定的 0.95 区间，原始 action 历史仍不裁剪。Preparation 继续使用自己的增益及物理限位，policy gains 不替代 FixStand gains。

## 验证结果

在独立物化的 `9e499a6` 源码上覆盖这两个文件，使用本地 Python 3.10 环境：

- 真实 `g1-largebox-a-upright-model_78000.deploy` 完整 SHA 校验、模型初始化及一次 CPU ONNX 推理通过；obs=(154,), action=(29,), 745 帧，clip=0.95。
- offline_check.py 禁止导入 Unitree/SDK，禁止创建网络 socket；只用内存假手柄和 RobotState，未创建 hardware runtime、DDS 或 LowCmd。
- 实测 settings 传递、Start/渐变/新 A 边沿、A 后仍处于 Preparation、模型尚未开始控制、正负极端 action 裁剪、旧 validate_for 和未知配置拒绝均通过。
- 旧版 test_g1_startup、test_g1_wbt_task、test_task_plugins、test_runtime、test_policy_controller、test_hardware_contracts、test_native_backend、test_g1_29dof_backend：124 passed。硬件相关测试使用 fake adapter/backend。
- 首次额外测试的子进程未继承隔离源码路径而错误加载本地新版 config；补齐 PYTHONPATH 后全部通过。

这是离线软件验收，不证明 PC2 的 SDK/native 二进制、网卡、固件或物理动作已验证。未启动真机部署。

## 上传及安装（只复制和离线检查）

本机（将 HOST 替换为实际 PC2 SSH 目标）：

```bash
scp g1-wbt-9e499a6.tar.gz HOST:/tmp/
```

PC2 在现有 MotrixLab 仓库根目录执行：

```bash
mkdir -p /tmp/g1-wbt-compat
 tar -xzf /tmp/g1-wbt-9e499a6.tar.gz -C /tmp/g1-wbt-compat
(cd /tmp/g1-wbt-compat && sha256sum -c SHA256SUMS)
backup_dir="/tmp/g1-wbt-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$backup_dir"
cp motrix_deploy_tasks/src/motrix_deploy_tasks/tasks/g1_wbt.py "$backup_dir/g1_wbt.py"
cp motrix_deploy/src/motrix_deploy/runtime/config.py "$backup_dir/config.py"
cp /tmp/g1-wbt-compat/g1_wbt.py motrix_deploy_tasks/src/motrix_deploy_tasks/tasks/g1_wbt.py
cp /tmp/g1-wbt-compat/config.py motrix_deploy/src/motrix_deploy/runtime/config.py
source .venv/bin/activate
python /tmp/g1-wbt-compat/offline_check.py --source-root "$PWD" --artifact artifacts/g1-largebox-a-upright-model_78000.deploy
```

无需运行部署命令完成此次安装验收。以后获准真机运行时，沿用旧版 `task=g1-wbt-dance/hardware`，覆盖 artifact、`runtime.transport.network_interface=<实际网卡>`、`duration_s=null`；保留 `task_settings.preparation.wait_for_remote_buttons=true`，不要套用新版 runtime.network_interface/robot 配置语法。
