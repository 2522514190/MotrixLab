# baseline 与 dr05 全指标对比

数据来自指定两次运行；训练终点均为 100000 轮。尾窗为最后约 1 万轮的 100 个日志点（异步记录的具体 step 可能略有偏移），标准差衡量时间波动，不是多随机种子置信区间。

共 100 个 scalar 标签；原始序列见 scalar_timeseries.csv，数值汇总见 all_metrics.csv。

训练超参数仅 task.env 名称不同。历史 source 快照显示：baseline 地面滑动摩擦为 1.0；dr05 为每条并行环境初始化采样 U(0.3,1.6)，reset 后保持不变；dr05 同时设置 floor.priority=2 并使用 GroundActionDrEnv 重写 reset。动作变化惩罚两者均为 -0.5。故实验同时改变了摩擦分布与接触材料优先级。

两组共用其他物理随机化、奖励、动作空间、参考动作、课程和终止条件。当前源码在训练后有修改；历史实验逻辑以 runs/g1-ground-action-100k-20261111-224839/source 和 tracked-source.patch 为准。

成功定义：从第 0 帧开始，持续抬箱并稳定放下，参考轨迹结束时位置误差 <0.15m、姿态误差 <0.30rad 且无失败终止，每回合最多计数一次。box_success_total 是累计事件数；box_success 是日志时刻的单步成功占比；lift/setdown_verified 是当前环境标记比例，均不能直接解释为回合成功率。现存日志没有完整起点回合总数，不能算严格成功率。

## 全部指标

| 指标 | baseline 最终 | dr05 最终 | baseline 尾窗均值 | dr05 尾窗均值 | 尾窗变化 |
|---|---:|---:|---:|---:|---:|
| async/policy_lag | 0 | 0 | 0 | 0 | — |
| async/ring_fill | 0 | 0 | 0 | 0 | — |
| async/utd | 3.996 | 3.996 | 3.999679 | 3.999571 | -0.00% |
| async/weight_publishes_per_iter | 0 | 0 | 1.001358 | 0.9884679 | -1.29% |
| async/weight_version | 99901 | 99901 | 95049 | 95048 | -0.00% |
| metrics/adaptive_failure_mass | 4.391621 | 4.281149 | 4.281515 | 4.298805 | +0.40% |
| metrics/adaptive_sampling_entropy | 0.9244811 | 0.8963355 | 0.9125905 | 0.8912721 | -2.34% |
| metrics/adaptive_sampling_top1_bin | 0.3333333 | 0.3333333 | 0.3333333 | 0.3333333 | +0.00% |
| metrics/adaptive_sampling_top1_prob | 0.1375126 | 0.1387389 | 0.1370476 | 0.1417571 | +3.44% |
| metrics/bad_body_z | 0 | 0.0004882812 | 0.0007617188 | 0.0007519531 | -1.28% |
| metrics/bad_dof_pos | 0 | 0 | 0 | 0 | — |
| metrics/bad_dof_vel | 0 | 0 | 0 | 0 | — |
| metrics/bad_ref_ori | 0 | 0 | 0 | 4.882813e-06 | — |
| metrics/bad_ref_z | 0 | 0 | 0 | 4.882813e-06 | — |
| metrics/body_z_max_err | 0.06937708 | 0.07858993 | 0.07014035 | 0.07180011 | +2.37% |
| metrics/box_curriculum_enabled | 1 | 1 | 1 | 1 | +0.00% |
| metrics/box_curriculum_survival_ratio | 0.8030631 | 0.7876157 | 0.805412 | 0.7905301 | -1.85% |
| metrics/box_curriculum_trigger_iteration | 16700 | 16988 | 16700 | 16988 | +1.72% |
| metrics/box_full_start_fraction | 0.4438477 | 0.4423828 | 0.4427051 | 0.4370801 | -1.27% |
| metrics/box_lift_verified | 0.01855469 | 0.03027344 | 0.0346875 | 0.04307617 | +24.18% |
| metrics/box_reward_weight | 0.6010775 | 0.5989951 | 0.6123442 | 0.5989951 | -2.18% |
| metrics/box_setdown_verified | 0.01123047 | 0.01904297 | 0.01747559 | 0.02424316 | +38.73% |
| metrics/box_success | 0 | 0 | 7.324219e-05 | 3.90625e-05 | -46.67% |
| metrics/box_success_total | 7754 | 6677 | 7365.23 | 6036.71 | -18.04% |
| metrics/clip_ended | 0.001464844 | 0.001464844 | 0.00137207 | 0.001235352 | -9.96% |
| metrics/dof_limit_violation_max | 0.0007700343 | 0.0007231657 | 0.0006013136 | 0.0006323732 | +5.17% |
| metrics/dof_vel_abs_max | 2.661758 | 2.612536 | 2.644594 | 2.640833 | -0.14% |
| metrics/motion_step | 392.9897 | 386.4756 | 387.785 | 384.9739 | -0.72% |
| metrics/object_orientation_error | 0.1545215 | 0.1480422 | 0.1523083 | 0.1503503 | -1.29% |
| metrics/object_position_error | 0.1635512 | 0.1628526 | 0.1574477 | 0.162556 | +3.24% |
| metrics/object_termination_enabled | 1 | 1 | 1 | 1 | +0.00% |
| metrics/object_tracking | 0.0004882812 | 0.002929688 | 0.0009716797 | 0.0009521484 | -2.01% |
| metrics/ref_ori_abs_err | 0.05119587 | 0.05453066 | 0.05201934 | 0.05310118 | +2.08% |
| metrics/ref_z_abs_err | 0.04084079 | 0.03706181 | 0.03836642 | 0.03752309 | -2.20% |
| metrics/tracking/body_angular_velocity_rms_rad_s | 1.7108 | 1.687898 | 1.706612 | 1.705168 | -0.08% |
| metrics/tracking/body_linear_velocity_rms_m_s | 0.3233814 | 0.3156435 | 0.3201641 | 0.3181392 | -0.63% |
| metrics/tracking/body_relative_orientation_rms_rad | 0.2699065 | 0.2567299 | 0.2735424 | 0.2536639 | -7.27% |
| metrics/tracking/body_relative_position_rms_m | 0.07735427 | 0.07684802 | 0.07641893 | 0.07547292 | -1.24% |
| metrics/tracking/joint_position_rms_rad | 0.2132598 | 0.2084569 | 0.2246254 | 0.2047362 | -8.85% |
| metrics/tracking/joint_velocity_rms_rad_s | 0.9886481 | 0.968563 | 0.990892 | 0.977597 | -1.34% |
| metrics/tracking/object_angular_velocity_error_rad_s | 0.6630291 | 0.6800105 | 0.6860081 | 0.6632216 | -3.32% |
| metrics/tracking/object_linear_velocity_error_m_s | 0.1759822 | 0.1761751 | 0.1809894 | 0.1779638 | -1.67% |
| metrics/tracking/reference_orientation_error_rad | 0.1318016 | 0.1500072 | 0.1403489 | 0.1438774 | +2.51% |
| metrics/tracking/reference_position_error_m | 0.1997031 | 0.1985363 | 0.1943916 | 0.1985381 | +2.13% |
| metrics/tracking_metrics | 0 | 0 | 0 | 0 | — |
| perf/collect_ms_per_batch | 84.94165 | 69.17062 | 83.55862 | 68.97474 | -17.45% |
| perf/collector_bookkeep_ms | 0.2934533 | 0.3099954 | 0.2946929 | 0.3120438 | +5.89% |
| perf/collector_env_step.apply_action_ms | 0.9841288 | 0.3677768 | 0.9296858 | 0.4136562 | -55.51% |
| perf/collector_env_step.final_obs_ms | 2.234326 | 1.235111 | 2.213599 | 1.32405 | -40.19% |
| perf/collector_env_step.observation_ms | 2.230866 | 1.369043 | 2.201752 | 1.382725 | -37.20% |
| perf/collector_env_step.physics_ms | 68.10551 | 59.39584 | 68.39236 | 59.21618 | -13.42% |
| perf/collector_env_step.reset_ms | 3.189619 | 0.9672048 | 2.63216 | 0.9837259 | -62.63% |
| perf/collector_env_step.transition_ms | 5.921268 | 3.336778 | 5.328263 | 3.388295 | -36.41% |
| perf/collector_env_step_ms | 82.20361 | 66.61803 | 81.10144 | 66.55022 | -17.94% |
| perf/collector_push_ms | 0.6379993 | 0.5765659 | 0.5585172 | 0.5638165 | +0.95% |
| perf/collector_sample_actions_ms | 1.528667 | 1.435951 | 1.372425 | 1.318048 | -3.96% |
| perf/collector_sync.actor_load_ms | 0.07774406 | 0.07961445 | 0.07645432 | 0.07929503 | +3.72% |
| perf/collector_sync.host_snapshot_ms | 0.1272499 | 0.07849621 | 0.08269406 | 0.07974753 | -3.56% |
| perf/collector_sync.wait_writer_ms | 0 | 0 | 0 | 0 | — |
| perf/collector_sync_ms | 0.2279433 | 0.1791992 | 0.1819627 | 0.1801592 | -0.99% |
| perf/collector_wait_ms | 0 | 0 | 0 | 0 | — |
| perf/env_steps_per_s | 24079.98 | 29574.28 | 24518.39 | 29679.43 | +21.05% |
| perf/learn_ms_total | 25.3374 | 25.13676 | 25.04595 | 25.24105 | +0.78% |
| perf/learn_pct | 30.84326 | 36.23579 | 29.98346 | 36.58241 | +22.01% |
| perf/updates_per_s | 0 | 0 | 47.94039 | 57.29973 | +19.52% |
| reward/action_rate_l2 | -0.01794498 | -0.01787302 | -0.0179716 | -0.01814214 | -0.95% |
| reward/box_success_bonus | 0.0004394531 | 0.0005371094 | 0.0004594727 | 0.0006499023 | +41.45% |
| reward/limits_dof_pos | -0.001683701 | -0.001434165 | -0.00178847 | -0.001509458 | +15.60% |
| reward/motion_global_body_ang_vel | 0.01472659 | 0.01480981 | 0.01472837 | 0.01473396 | +0.04% |
| reward/motion_global_body_lin_vel | 0.01763589 | 0.01773879 | 0.01766584 | 0.01770302 | +0.21% |
| reward/motion_global_ref_orientation_error_exp | 0.008752237 | 0.008528079 | 0.008649943 | 0.008586005 | -0.74% |
| reward/motion_global_ref_position_error_exp | 0.006576227 | 0.006577006 | 0.006682137 | 0.006585387 | -1.45% |
| reward/motion_relative_body_orientation_error_exp | 0.01297915 | 0.0133704 | 0.01259615 | 0.01333973 | +5.90% |
| reward/motion_relative_body_position_error_exp | 0.01845284 | 0.0185017 | 0.01844875 | 0.0184892 | +0.22% |
| reward/object_global_ref_orientation_error_exp | 0.009878201 | 0.00992575 | 0.01011207 | 0.00991195 | -1.98% |
| reward/object_global_ref_position_error_exp | 0.008466141 | 0.008492595 | 0.008855365 | 0.00854508 | -3.50% |
| reward/undesired_contacts | -0.0008703223 | -0.001058135 | -0.0009223922 | -0.001135766 | -23.13% |
| rollout/mean_ep_len | 389.36 | 336.88 | 405.9273 | 405.8563 | -0.02% |
| rollout/mean_return | 30.55325 | 25.75333 | 31.49194 | 31.48008 | -0.04% |
| train/actor_loss | -6.774345 | -6.700161 | -6.798382 | -6.708415 | +1.32% |
| train/alpha | 0.0007244878 | 0.0007592652 | 0.0007281051 | 0.0007707509 | +5.86% |
| train/alpha_loss | 3.586493e-05 | 5.822157e-06 | 1.388552e-05 | 1.105524e-05 | -20.38% |
| train/alpha_loss_abs | 9.703993e-05 | 9.449533e-05 | 9.20398e-05 | 9.466048e-05 | +2.85% |
| train/alpha_loss_abs_max | 0.0003610428 | 0.0003097388 | 0.0003289607 | 0.0003356451 | +2.03% |
| train/alpha_loss_abs_min | 3.567012e-07 | 6.175123e-07 | 4.089707e-07 | 4.039679e-07 | -1.22% |
| train/batch_reward_max | 3.307056 | 4.131134 | 3.241508 | 4.214065 | +30.00% |
| train/batch_reward_max_max | 10.10769 | 10.10726 | 10.10672 | 10.10992 | +0.03% |
| train/batch_reward_max_min | 0.1118999 | 0.1113999 | 0.1120757 | 0.1116301 | -0.40% |
| train/batch_reward_mean | 0.07802742 | 0.07760153 | 0.07760518 | 0.07773537 | +0.17% |
| train/final_obs_events | 0.84 | 0.81 | 0.7628203 | 0.7798403 | +2.23% |
| train/policy_entropy | -14.46411 | -14.45876 | -14.48936 | -14.48437 | +0.03% |
| train/policy_entropy_max | -14.01143 | -14.05147 | -14.10287 | -14.10862 | -0.04% |
| train/policy_entropy_min | -14.75753 | -14.85274 | -14.85691 | -14.85308 | +0.03% |
| train/qf_loss | 6.12539 | 6.233503 | 6.172199 | 6.217964 | +0.74% |
| train/qf_max | 9.468636 | 9.51649 | 9.508386 | 9.522074 | +0.14% |
| train/qf_max_max | 10.10769 | 10.10726 | 10.10672 | 10.10992 | +0.03% |
| train/qf_max_min | 9.151506 | 9.096609 | 9.221515 | 9.087289 | -1.46% |
| train/qf_min | -1.45746 | -1.107839 | -1.285029 | -1.117797 | +13.01% |
| train/qf_min_max | -0.9010308 | -0.6372727 | -0.7412306 | -0.6016685 | +18.83% |
| train/qf_min_min | -2.188917 | -2.304148 | -2.143096 | -1.974212 | +7.88% |

## 检查点

两者 latest.pt 均为 global_step=100000、update_idx=399996；模型形状相同，学习权重不同；manifest 的 best_policy 也指向 latest.pt，不能当作独立选优模型。每组 99 个 model_*.pt（1000–99000，间隔 1000）和 latest.pt。

## 限制

这是各自训练分布下、单个种子的训练日志比较。未重新运行统一地面摩擦、关闭探索的评估，不能据此判定鲁棒性或实际部署成功率。速度差来自实际计时，但不能只归因于地面随机化。两组正常完成（exit_code=0），结束均有 CUDA IPC tensor 释放警告。

## 共同训练设置与运行差异

- 两份 task_config.yaml 除 task.env 外逐字段相同；每组 seed=1、2048 环境、100000 轮、204800000 环境步；checkpoint 的 update_idx 均为 399996。
- FastSAC 异步，1 collector / 1 learner；strict UTD，4 updates/轮；actor/critic/alpha 学习率均 3e-4，batch=8192，buffer=1024，gamma=0.99，tau=0.05，policy_frequency=2。
- actor_hidden_dim=512、critic_hidden_dim=768、2 个 Q 网络、501 atoms、值域 [-20,20]；obs normalization、LayerNorm、tanh、torch.compile 均开启。learner 使用 bf16 AMP，collector 使用 fp16 AMP。
- 最终 checkpoint 验证 actor 输入 154、输出 29，critic_obs_normalizer 维度 298；两者动作 scale/bias 相同，所有组件张量形状相同，学习权重与归一化统计不同。
- 同一 RTX 5070 Ti、Ryzen 7 9700X、16 逻辑核；Python 3.10.21、Torch 2.11.0、MotrixSim 0.10.1。队列记录同一 commit 49e7cf168c8c63ac64cd45f9af93ab8e52e28016，但队列还保存了未提交源码快照。
- baseline 创建于北京时间 11月11日22:48:40，11月12日01:23:22 保存最终模型；dr05 为 11月12日03:09:28 至05:01:36，顺序运行。
- metadata 训练耗时：baseline 9283.1s，dr05 6728.9s；平均吞吐：22061 vs 30435 环境步/s。
- baseline 环境启动 65.752s，dr05 为 0.293s；baseline evaluate/observe specialization 为38.694/26.652s，dr05 为0.017/0.013s。启动缓存差异只解释约65s，不能解释全部2554.2s耗时差。
- 尾窗采集耗时83.559 vs68.975ms，物理68.392 vs59.216ms，reset 2.632 vs0.984ms，transition 5.328 vs3.388ms；learner 为25.046 vs25.241ms。主要速度差出现在环境采集侧。
- 两组都开启箱子质量、质心、惯量、箱子摩擦、躯干质心及外部推扰随机化。baseline 仅表示没有新增地面随机化，不代表没有其他域随机化。
- 两组动作变化惩罚 -0.5、非期望接触惩罚 -0.1，开启自适应起点采样与延迟物体失败终止；同一745帧参考动作，50Hz控制、16s回合上限。

## 关键结果解释

累计完整成功为7754 vs6677，dr05少1077次（-13.89%）。但最后约1万轮100点均值的回报为31.49194 vs31.48008（-0.038%），回合长度405.9273 vs405.8563步（-0.017%）。最终单点回报30.55325 vs25.75333，单点波动不代表整个尾窗水平。

尾窗物体位置误差0.157448 vs0.162556m（dr05增加3.24%）；姿态误差0.152308 vs0.150350rad（减少1.29%）；关节位置RMS 0.224625 vs0.204736rad（减少8.85%）。抬箱标记占比3.46875% vs4.30762%，放箱标记占比1.74756% vs2.42432%；这些是环境当前状态占比，不是回合成功率。

本比较已读取两次运行全部100个scalar标签；未发现非有限数值。完整原始值、末点、尾窗均值、标准差、全程最小最大值及变化百分比均在CSV中。

按90000轮附近最近的日志计数至100000轮，baseline新增约941次成功、dr05约1351次。dr05后期新增成功更多，说明累计总数差距不能解释为后期仍持续落后；由于异步日志采样点略有偏移，这不是完全对齐的精确区间比较。
