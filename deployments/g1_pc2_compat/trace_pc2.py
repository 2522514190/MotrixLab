"""Diagnostic launcher: observe existing PC2 calls without changing control behavior."""
import faulthandler
import functools
import os
import signal
import sys
import time

faulthandler.enable(all_threads=True)
if hasattr(signal, 'SIGUSR1'):
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', file=sys.stderr, flush=True)


log(f'诊断启动，PID={os.getpid()}；尚未连接机器人。')
log('导入部署框架…')
import motrix_deploy.cli as cli
from motrix_deploy.policy import OnnxPolicy
from motrix_deploy.runtime.control import ControlSession
from motrix_deploy_pc2 import wbt_box, wbt_robot_only


def trace_call(original, label):
    @functools.wraps(original)
    def call(*args, **kwargs):
        log(label + '：开始')
        try:
            result = original(*args, **kwargs)
        except BaseException as error:
            log(label + f'：退出，{type(error).__name__}: {error}')
            raise
        log(label + '：完成')
        return result
    return call


cli.read_artifact = trace_call(cli.read_artifact, '读取并校验 artifact')
OnnxPolicy.__init__ = trace_call(OnnxPolicy.__init__, 'ONNX 初始化')
_original_factory = cli.create_hardware_runtime


def hardware_factory(*args, **kwargs):
    runtime = trace_call(_original_factory, '构造硬件 runtime')(*args, **kwargs)
    for name, label in [('open', '硬件 open/连接'), ('acquire_control', '获取低层控制权'),
                        ('enable_control', '使能控制'), ('stop', '硬件停止/阻尼'), ('close', '关闭硬件')]:
        if hasattr(runtime.robot, name):
            setattr(runtime.robot, name, trace_call(getattr(runtime.robot, name), label))
    original_close = runtime.robot.close
    def close_with_stack_watchdog(*args, **kwargs):
        log('开始硬件清理；若超过 5 秒，将自动打印线程栈，不需要发送信号')
        faulthandler.dump_traceback_later(5, repeat=False)
        try:
            return original_close(*args, **kwargs)
        finally:
            faulthandler.cancel_dump_traceback_later()
    runtime.robot.close = close_with_stack_watchdog
    return runtime


cli.create_hardware_runtime = hardware_factory

_original_fail = ControlSession.fail
_original_stop = ControlSession.stop


def session_fail(self, error):
    log(f'控制循环异常（清理前记录）：{type(error).__name__}: {error}')
    return _original_fail(self, error)


def session_stop(self):
    log(f'退出原因（清理前）：{getattr(self, "_reason", None)}；'
        f'详情={getattr(self, "_message", None)}；'
        f'策略周期={self.completed_steps}；'
        f'播放帧={getattr(self.controller, "playback_steps", None)}')
    return _original_stop(self)


ControlSession.fail = session_fail
ControlSession.stop = session_stop

messages = {
    'wait_stand': '等待 Start（此阶段尚未执行站姿过渡）',
    'stand': 'Start 已接受，正在过渡到默认站姿；先不要按 A',
    'wait_policy': '等待新的 A 按键；确认站稳后再按',
    'prepare': '执行策略接管前的实测准备检查',
    'policy_hold': '策略已接管，冻结首帧参考',
    'playback': '正在播放参考动作',
    'final_hold': '正在保持末帧，策略仍持续推理',
    'damping': '进入阻尼停止',
    'complete': '任务结束',
}


def instrument(cls):
    cls.prepare = trace_call(cls.prepare, '装配 Start/A 启动流程')
    original = cls.step
    last = [None]
    last_log = [0.0]

    def report(task):
        startup = getattr(getattr(task, '_startup', None), 'phase', None)
        phase = startup if startup and startup != 'complete' else task.phase
        now = time.monotonic()
        if phase != last[0] or now - last_log[0] >= 10:
            log('控制阶段：' + messages.get(phase, str(phase)))
            last[0], last_log[0] = phase, now

    @functools.wraps(original)
    def step(self, *args, **kwargs):
        report(self)
        result = original(self, *args, **kwargs)
        report(self)
        return result
    cls.step = step


instrument(wbt_box.G1WbtDeployTask)
instrument(wbt_robot_only.G1WbtDeployTask)
log('框架已加载，进入原 CLI。--cfg job 仅打印配置；正常启动参数会连接实机。')
cli.main()
