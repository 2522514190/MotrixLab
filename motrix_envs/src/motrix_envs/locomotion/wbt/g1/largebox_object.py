# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""G1 robot and dynamic-box reference tracking."""

from __future__ import annotations

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.mdp.rewards import ActionRateRewardCfg
from motrix_env_core.sim import (
    LinkAngularVelocityQuery,
    LinkLinearVelocityQuery,
    LinkPositionQuery,
    LinkQuaternionQuery,
)
from motrix_envs.locomotion.wbt.cfg import ObservationsCfg, ResetCfg, RewardsCfg, TerminationsCfg
from motrix_envs.locomotion.wbt.g1.common import MOTION_DIR, G1WbtEnvCfg, G1WbtSceneObjsCfg, make_large_box
from motrix_envs.locomotion.wbt.mdp.object import (
    OBJECT_CHANNELS,
    ObjectLinearVelocityObsCfg,
    ObjectMassRandomizationCfg,
    ObjectOrientationObsCfg,
    ObjectOrientationRewardCfg,
    ObjectPositionObsCfg,
    ObjectPositionRewardCfg,
    ObjectResetCfg,
    ObjectTrackingTerminationCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    DofLimitRewardCfg,
    GlobalBodyAngularVelocityRewardCfg,
    GlobalBodyLinearVelocityRewardCfg,
    GlobalRefOrientationRewardCfg,
    GlobalRefPositionRewardCfg,
    RelativeBodyOrientationRewardCfg,
    RelativeBodyPositionRewardCfg,
    UndesiredContactsRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.terminations import BadBodyZTerminationCfg
from motrix_envs.locomotion.wbt.mdp.tracking_metrics import LargeBoxTrackingMetricsCfg


@configclass
class LargeBoxRewardsCfg(RewardsCfg):
    # Keep all task reward parameters here; do not override them in the factory below.
    # Robot terms match MotrixLab G1 WBT; object terms follow Holosoma w_object.
    # 正奖励权重越大，越重视该跟踪目标；负权重绝对值越大，惩罚越强。
    # 指数跟踪项为 exp(-平均平方误差 / sigma²)；sigma 越小越严格，但大误差时奖励更易接近零。

    # 参考 body 的世界坐标位置跟踪：约束机器人整体位置和高度；sigma 单位为 m。
    motion_global_ref_position_error_exp: GlobalRefPositionRewardCfg = GlobalRefPositionRewardCfg(
        weight=1.0,
        sigma=0.3,
    )
    # 参考 body 的世界坐标朝向跟踪：约束整体倾斜和转向；sigma 单位为 rad。
    motion_global_ref_orientation_error_exp: GlobalRefOrientationRewardCfg = GlobalRefOrientationRewardCfg(
        weight=0.5,
        sigma=0.4,
    )
    # 各跟踪 body 到对齐后的参考位置的平均误差：约束四肢与躯干的动作形状；sigma 单位为 m。
    motion_relative_body_position_error_exp: RelativeBodyPositionRewardCfg = RelativeBodyPositionRewardCfg(
        weight=2.0,
        sigma=0.2,
    )
    # 各跟踪 body 到对齐后的参考朝向的平均误差：约束身体各部位的旋转姿态；sigma 单位为 rad。
    motion_relative_body_orientation_error_exp: RelativeBodyOrientationRewardCfg = RelativeBodyOrientationRewardCfg(
        weight=1.0,
        sigma=0.4,
    )
    # 各跟踪 body 的世界坐标线速度跟踪：约束运动方向和快慢；sigma 单位为 m/s。
    motion_global_body_lin_vel: GlobalBodyLinearVelocityRewardCfg = GlobalBodyLinearVelocityRewardCfg(
        weight=1.0,
        sigma=1.0,
    )
    # 各跟踪 body 的世界坐标角速度跟踪：约束旋转方向和快慢；sigma 单位为 rad/s。
    motion_global_body_ang_vel: GlobalBodyAngularVelocityRewardCfg = GlobalBodyAngularVelocityRewardCfg(
        weight=1.0,
        sigma=3.14,
    )
    # 惩罚相邻控制步动作差的平方和，抑制指令抖动；惩罚过强可能妨碍快速跟踪。
    action_rate_l2: ActionRateRewardCfg = ActionRateRewardCfg(weight=-0.5)
    # 惩罚关节位置超出软限位的幅度总和；soft_limit=0.9 表示保留关节范围中央的 90%。
    limits_dof_pos: DofLimitRewardCfg = DofLimitRewardCfg(
        weight=-10.0,
        soft_limit=0.9,
        # 原始越界惩罚封顶为 5，再乘 weight，避免失稳环境主导 Q 学习目标。
        cap=5.0,
    )
    # 惩罚被配置为非期望接触的 body 数量；接触力大于 threshold（N）才计数，并非惩罚所有接触。
    undesired_contacts: UndesiredContactsRewardCfg = UndesiredContactsRewardCfg(
        weight=-0.1,
        threshold=1.0,
    )
    # 箱子世界坐标位置跟踪：约束箱子按同一参考帧移动、抬起和放下；sigma 单位为 m。
    object_global_ref_position_error_exp: ObjectPositionRewardCfg = ObjectPositionRewardCfg(
        weight=1.0,
        sigma=0.3,
    )
    # 箱子世界坐标朝向跟踪：约束箱子的倾斜和旋转；sigma 单位为 rad。
    object_global_ref_orientation_error_exp: ObjectOrientationRewardCfg = ObjectOrientationRewardCfg(
        weight=1.0,
        sigma=0.4,
    )


@configclass
class LargeBoxTerminationsCfg(TerminationsCfg):
    bad_body_z: BadBodyZTerminationCfg = BadBodyZTerminationCfg(
        threshold=0.25,
        body_names=("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"),
    )
    object_tracking: ObjectTrackingTerminationCfg = ObjectTrackingTerminationCfg()
    # 仅记录原始跟踪误差，始终返回 False，不触发终止。
    tracking_metrics: LargeBoxTrackingMetricsCfg = LargeBoxTrackingMetricsCfg()


@configclass
class LargeBoxResetCfg(ResetCfg):
    object_state: ObjectResetCfg = ObjectResetCfg(body="baseLink")
    # Training mass (kg), sampled once per environment and retained across resets.
    object_mass: ObjectMassRandomizationCfg = ObjectMassRandomizationCfg(
        body="baseLink",
        light_probability=1.0,
        light_range=(0.1, 0.1),
        heavy_range=(0.1, 0.1),
        # Inertia = mass * this coefficient: 0.1 kg gives diagonal 0.002 kg m^2.
        inertia_per_mass=(0.02, 0.02, 0.02, 0.0, 0.0, 0.0),
    )


@configclass
class LargeBoxValueCfg(ObservationsCfg.ValueCfg):
    obj_pos_b: ObjectPositionObsCfg = ObjectPositionObsCfg()
    obj_ori_b: ObjectOrientationObsCfg = ObjectOrientationObsCfg()
    obj_lin_vel_b: ObjectLinearVelocityObsCfg = ObjectLinearVelocityObsCfg()


@configclass
class LargeBoxObservationsCfg(ObservationsCfg):
    value: LargeBoxValueCfg = LargeBoxValueCfg()


@configclass
class G1LargeBoxEnvCfg(G1WbtEnvCfg):
    rewards: LargeBoxRewardsCfg = LargeBoxRewardsCfg()
    terminations: LargeBoxTerminationsCfg = LargeBoxTerminationsCfg()
    sim_reset: LargeBoxResetCfg = LargeBoxResetCfg()
    observations: LargeBoxObservationsCfg = LargeBoxObservationsCfg()

    def for_play(self) -> G1LargeBoxEnvCfg:
        cfg = super().for_play()
        cfg.sim_reset.object_state.noise_scale = 0.0
        cfg.sim_reset.object_mass.light_range = (0.1, 0.1)
        cfg.sim_reset.object_mass.heavy_range = (0.1, 0.1)
        return cfg


@registry.envcfg("g1-29dof-wbt-largebox-object")
def make_g129dof_wbt_largebox_object_cfg() -> G1LargeBoxEnvCfg:
    """Track synchronized G1 and box references with physical object interaction.

    zh_CN: 让 Unitree G1 通过物理接触搬运箱子，同步跟踪机器人与箱子的参考动作。
    """
    cfg = G1LargeBoxEnvCfg(motion_files=(str(MOTION_DIR / "sub3_largebox_003_w_obj.npz"),))
    cfg.scene.objs = G1WbtSceneObjsCfg(robot=cfg.scene.objs.robot, box=make_large_box())
    cfg.commands.motion.extension_channels = OBJECT_CHANNELS
    body = cfg.scene.objs.box.resolved_base_link_name
    cfg.sim_reset.object_state.body = body
    cfg.sim_reset.object_mass.body = body
    cfg.queries.data.update(
        object_pos=LinkPositionQuery(link=body),
        object_quat=LinkQuaternionQuery(link=body),
        object_lin_vel=LinkLinearVelocityQuery(link=body),
        object_ang_vel=LinkAngularVelocityQuery(link=body),
    )
    return cfg


registry.env("g1-29dof-wbt-largebox-object")(ManagerEnv)
