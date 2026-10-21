# OmniRetarget 搬箱 PPO / FastSAC：参数来源与复现边界

本配置使用 MotrixLab 现有的 RSLRL PPO provider、ManagerEnv、sim read/write 和
ContactSensorCfg 接口。新增内容属于任务配置和任务 MDP；不修改 core、通用 backend
或 PPO provider。已有 SAC 任务保持独立。

参数依据依次为 OmniRetarget 第 IV 节、其引用的 BeyondMimic 补充材料，以及论文未说明
之处的 Holosoma WBT PPO 配置。仓库里已经调过的 SAC 奖励和训练参数不是本配置的依据。
这是一套按上述来源建立的训练基线，尚未验证长期训练效果或硬件零样本迁移。

## 来源

- `/home/yudong-chen/lunwen/OmniRetarget.pdf`，IV：观测、五类奖励、物体终止和随机化范围。
- `/home/yudong-chen/lunwen/BeyondMimic.pdf`，S1、Table S1、Table S5：奖励公式、权重、sigma、
  执行器增益、动作缩放、身体终止、自适应参考采样、PPO 网络和优化超参数。
- `/home/yudong-chen/holosoma`，读取时 HEAD 为
  `b633c2fcbcaf33b68ffd25e2da6f0c8e907266cb`。
  `src/holosoma/holosoma/config_values/wbt/g1/` 中的 `experiment.py`、`observation.py`、
  `reward.py`、`command.py`、`randomization.py`、`termination.py`，以及
  `config_values/algo.py`、`config_values/robot.py`。

## PPO 与控制

| 项目 | 设置 | 来源 |
| --- | --- | --- |
| Actor / critic | 独立 MLP，512 / 256 / 128，ELU | BeyondMimic Table S5 |
| Rollout / iterations | 24 / 30000 | BeyondMimic Table S5 |
| Learning rate / epochs / mini-batches | 0.001 / 5 / 4 | BeyondMimic Table S5 |
| Clip / entropy / value loss | 0.2 / 0.005 / 1.0 | BeyondMimic Table S5 |
| Gamma / GAE lambda / desired KL | 0.99 / 0.95 / 0.01 | BeyondMimic Table S5 |
| Adaptive LR / gradient clip | adaptive / 1.0 | Holosoma PPO |
| Normalization / initial action std | running normalization / 1.0 | Holosoma WBT PPO |
| Optimizer | Adam；Holosoma 为 weight_decay=0 的 AdamW | MotrixLab RSLRL 的对应实现 |
| Parallel envs / checkpoint interval | 4096 / 4000 | Holosoma WBT PPO |
| Episode horizon / sim / control | 10 s / 200 Hz / 50 Hz | Holosoma WBT |
| Symmetry / RND | 关闭 | Holosoma WBT PPO / 不额外添加机制 |

BeyondMimic S1 的控制参数为 `kp = I * (2π*10)^2`、`kd = 4*I*(2π*10)`，
`action_scale = 0.25 * torque_limit / kp`。任务显式配置这些值，使用 G1 的对应 armature
和力矩上限；位置目标的 `ctrl_range=(-inf, inf)` 直接通过已有配置接口声明，机械关节
限位仍由机器人模型保留。关节默认姿态与 Holosoma `robot.py` 的 G1 默认姿态一致。

Holosoma 的初始回合进度随机化通过本任务的 `init_state()` 后处理现有 `episode_steps`
实现，不增加 provider 字段，不改变 reset/step 生命周期。该随机进度不参与物体终止课程
的“实际存活步数”统计。

## 观测与奖励

搬箱保留 pelvis 位置误差和线速度。Actor 依次接收参考关节位置/速度、参考 pelvis
位置误差与 Rot6D 朝向误差、pelvis 线/角速度、校准后的关节位置、关节速度和上一动作，
共 160 维；没有实测箱子或场景输入。Rot6D 使用旋转矩阵前两列，按行展开，与 Holosoma
实现一致。Critic 增加身体相对位姿和箱子的位置、朝向、线速度，共 298 维，关闭观测噪声。

| 奖励项 | weight | sigma |
| --- | --- | --- |
| Anchor 位置 / 朝向 | 0.5 / 0.5 | 0.3 / 0.4 |
| 相对身体位置 / 朝向 | 1.0 / 1.0 | 0.3 / 0.4 |
| 身体线速度 / 角速度 | 1.0 / 1.0 | 1.0 / 3.14 |
| 箱子位置 / 朝向 | 1.0 / 1.0 | 0.3 / 0.4 |
| 动作差平方和 | -0.1 | — |
| 软关节限位 | -10.0 | 软范围 0.9，无额外惩罚截断 |
| 自碰撞 | -0.1 | 非末端身体净自接触力 > 1 N 时按身体计数 |

跟踪项为 `exp(-mean_squared_error / sigma^2)`。权重按照 MotrixLab 已有 reward manager
的规则乘控制时间步。自碰撞使用现有几何对接触传感器汇总机器人内部接触力，因此地面和箱子
接触不会进入该惩罚。传感器的力分量根据所报告的 normal/tangent 从接触坐标系转到世界
坐标系，再对同一身体的多个接触按向量汇总，不重复计数。

## 终止、参考采样与随机化

身体终止采用 BeyondMimic S1：pelvis 或任一腕/踝垂直位置误差 > 0.25 m，或 pelvis
完整四元数朝向误差 > 0.8 rad。物体终止采用 OmniRetarget：位置误差 > 1 m 或朝向误差
> 45°，并在身体跟踪形成后才启用。

**启用门槛不是论文给出的数值。** 两篇论文没有明确“reasonable body tracking”的量化条件，
所读取的 Holosoma WBT 代码也没有延后启用物体终止的 gate。因此采用其 episode-length
EMA（窗口参数 1000），实际平均存活步数达到回合上限 80% 后锁存启用。
`object_curriculum` 暴露该复现选择；显式 reset 和参考 clip 重置不能单独触发课程。
播放/评估从一开始启用物体判据。

参考采用 BeyondMimic 的一秒分箱、failure EMA alpha=0.001、uniform floor=0.1、
三项前瞻核 `[1, 0.8, 0.8^2]`，由现有 WBT command 执行。

| 随机项 | 设置 | 时机 |
| --- | --- | --- |
| 箱子质量 | U(0.1, 2) kg | 每个环境启动时，沿用 Holosoma object DR 时机 |
| 箱子质心 | 每轴 ±0.08 m | 启动时，回合/clip reset 后保留 |
| 箱子惯量 | 名义惯量按质量比缩放，再乘 U(0.5, 1.5) | 启动时 |
| 躯干质心 | x/y/z ±0.025/0.05/0.075 m | 每回合重采样，参考 BeyondMimic S1 |
| 默认关节校准偏置 | ±0.01 rad，同时影响动作目标和位置观测 | 每回合重采样 |
| 推扰 | 线速度 ±0.3 m/s、角速度 ±0.78 rad/s，间隔 U(1, 3) s | 训练中 |
| Actor 观测噪声 | Rot6D ±0.05；线/角速度 ±0.5/0.2；关节位置/速度 ±0.01/0.5 | 每次观测 |

推扰标量在论文中没有分轴定义，本配置在三个世界坐标轴上分别采样并加到当前速度，
这是明确的实现选择。箱子惯量对三个主分量采用同一倍率；Holosoma 公共配置只随机 Ixx，
这里按 OmniRetarget 的物体惯量随机化描述处理。物体初始化沿用 Holosoma 的 XY 位置扰动
±0.05 m、参考线速度和零初始角速度；机器人初始化噪声来自其 WBT `command.py`。

## 当前差异与运行方式

**没有实现论文的箱子形状 ±10% 随机化。** MotrixLab 当前的通用接口没有按环境切换
缩放碰撞网格的操作。按本次“不修改底层”的约束，保留固定的原始箱子网格，并明确记录
此差异；不以视觉缩放代替碰撞形状变化，也不新增 backend。

默认只配置当前可用的一条 Holosoma 搬箱参考 `sub3_largebox_003_w_obj.npz`。论文的
“全部搬箱动作共享一个多任务策略”需要提供多条带物体轨迹的参考，再配置到
`commands.motion.motion_files`，现有 command 支持多 clip。单条默认参考不能称为论文的
完整多任务数据复现。仿真使用 MotrixSim，并不等同于论文/开源代码的原始仿真器。

```bash
source .venv/bin/activate
python scripts/train.py task=g1-29dof-wbt-largebox-omniretarget/rslrl.ppo
```

短训练验证可覆盖为 `num_envs=8 algo.max_iterations=2 checkpoint.interval=0`。
行为测试位于 `motrix_envs/tests/test_omniretarget.py`；通过短训练仅能确认训练链路运行，
不能证明已经学会搬箱或具备 sim-to-real 效果。

## 使用同一 Manager 任务的 FastSAC

`g1-29dof-wbt-largebox-omniretarget-fastsac` 的配置 factory 直接调用上述 PPO 配置
factory，运行时继承 `OmniRetargetEnv`。观测、奖励、终止课程、随机化、reset、参考采样和
PD 映射共用实现；原 PPO 注册名和配置保持不变。单独的注册名只用于 FastSAC 策略动作接口。

PPO 的无界高斯策略可以使用无界动作空间，而现有 FastSAC provider 从动作空间的有限
上下界计算 tanh 缩放。直接使用 PPO 的无界空间会得到无限缩放和 NaN 偏置。因此任务层
变体按 Holosoma `FastSACEnvWrapper._compute_action_boundaries` 提供对称策略边界：
`max(abs(joint_lower - default), abs(joint_upper - default)) / action_scale`。
分母沿用其标量 `control.action_scale`，不是实际 PD 映射中的逐关节 `effort/kp` 缩放。
该策略范围决定 FastSAC 的输出支持集；动作 term 和执行器位置目标仍无界，机械限位、
力矩限幅和校准偏置保持原样。未修改通用 core、backend 或 FastSAC provider。

算法参数来自 Holosoma `g1_29dof_wbt_fast_sac_w_object`，继承其 WBT FastSAC 和
`config_values/algo.py` 的 FastSAC 基础值。本地所有 30 个 agent 配置字段与对应值一致：
Actor/Critic/alpha 学习率均为 3e-4、batch size 8192、每环境 buffer 1024、1-step return、
每轮 4 次 Critic 更新、Actor 更新频率 2、gamma 0.99、tau 0.05、目标熵比例 0.5、
501 个 Q atoms、值域 [-20, 20]。框架执行采用已有异步 FastSAC 和 strict UTD 模式。
这里不使用 Holosoma 的 SAC 专用奖励覆盖，而是保留已经验证过的 PPO 环境奖励。

按用户要求使用 2048 个环境，训练上限 800000 轮，与 Holosoma 的 4096 × 400000
具有相同交互量。192000 轮对应本地 PPO 的 4096 × 24 × 4000 交互量。交互量相等不代表
更新次数、时间或效果相同：固定 batch size 和每轮更新数后，环境数减半会提高数据重复
使用比例。以上轮数均为预算/评估点，不能保证搬箱成功。

现有 FastSAC wrapper 保留 `terminated` 和 `truncated` 两个标志；buffer 会对超时继续
bootstrap。但该 provider 当前未单独存储 auto-reset 前的最终观测，超时处可能使用 reset
后的下一观测。此项是现有 provider 的边界，与 Holosoma 携带最终观测的路径存在差异；
本次任务层适配不改写回放池或通用 reset 生命周期。

```bash
source .venv/bin/activate
python scripts/train.py task=g1-29dof-wbt-largebox-omniretarget-fastsac/motrix.fastsac
```

任务与 FastSAC wrapper 的接线行为测试位于 `tests/test_omniretarget_fastsac.py`。

2026-10-21 验证：配置、任务行为和回放/训练选择测试共 157 项通过，另通过异步采样进程
测试。CPU 冒烟训练使用 8 个环境、32 轮、batch size 64、共享队列容量 2，保留默认网络和
算法其余设置；完成 116 次 Critic 更新并保存 checkpoint，网络/优化器状态及训练指标有限。
这些缩小参数仅用于命令行验证，没有写入正式训练配置；短跑不代表已经学会搬箱。
