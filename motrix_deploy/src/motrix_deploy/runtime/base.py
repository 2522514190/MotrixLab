# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Runtime resources, scheduling and physics advancement of a bound control session."""

import logging
from abc import ABC, abstractmethod
from dataclasses import fields, replace
from typing import TYPE_CHECKING

import numpy as np

from motrix_deploy.errors import ValidationError
from motrix_deploy.runtime.config import SimulationRandomizationCfg, VelocityPushCfg
from motrix_deploy.runtime.control import ControlSession
from motrix_deploy.runtime.result import RolloutResult, SimulationRolloutResult
from motrix_deploy.runtime.scheduler import LoopScheduler
from motrix_env_core.input import GamePadDevice, KeyboardDevice

if TYPE_CHECKING:
    from motrix_deploy.robot.interface import RobotInterface


class DeploymentRuntime(ABC):
    """Own world resources and scheduling, independent of controller semantics."""

    def __init__(self) -> None:
        self._opened = False
        self.session: ControlSession | None = None
        self._used = False

    def bind_session(self, session: ControlSession) -> None:
        if self.session is not None and self.session is not session:
            raise ValidationError("runtime.session", "the already bound control session", session)
        if session.robot is not self.robot:
            raise ValidationError("runtime.session.robot", self.robot, session.robot)
        self.session = session

    @property
    @abstractmethod
    def robot(self) -> "RobotInterface": ...

    @property
    def keyboard(self) -> KeyboardDevice | None:
        """Viewer keyboard input, or None when this runtime has no viewer."""
        return None

    @property
    def gamepad(self) -> GamePadDevice | None:
        """Operator gamepad input, or None when this runtime provides none."""
        return None

    @property
    def opened(self) -> bool:
        return self._opened

    def open(self) -> None:
        self._opened = True

    def close(self) -> None:
        self._opened = False

    def __enter__(self) -> "DeploymentRuntime":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _advance_interval(self) -> None:
        """Apply host-specific execution after an accepted command."""

    def _check_running(self) -> None:
        """Check whether the host permits another control interval."""

    def _run_session(self, scheduler: LoopScheduler) -> RolloutResult:
        if self.session is None:
            raise ValidationError("runtime.session", "a bound control session", None)
        if self._used:
            raise ValidationError("runtime.session", "a single-use execution", "already executed")
        self._used = True
        session = self.session
        try:
            if session.start():
                scheduler.reset()
                interval_step = 0
                while session.active:
                    self._check_running()
                    if not session.tick(elapsed_time_s=scheduler.elapsed_time_s(interval_step)):
                        break
                    self._advance_interval()
                    interval_step += 1
                    scheduler.wait(interval_step)
        except (Exception, KeyboardInterrupt) as error:
            session.fail(error)
        finally:
            session.stop()
        return session.result(overrun_count=scheduler.overrun_count)

    @abstractmethod
    def run(self) -> RolloutResult: ...


class SimulationRuntime(DeploymentRuntime, ABC):
    """Runtime that additionally owns simulation advancement and viewer resources."""

    def __init__(
        self, push: VelocityPushCfg | None = None, randomization: SimulationRandomizationCfg | None = None
    ) -> None:
        super().__init__()
        self._push = push or VelocityPushCfg()
        self._push_rng = np.random.default_rng(self._push.seed)
        self._randomization = randomization or SimulationRandomizationCfg()
        self._physical_rng = np.random.default_rng(self._randomization.seed)
        self._load_draw = None
        self._policy_simulation_time_s = 0.0

    def bind_session(self, session: ControlSession) -> None:
        if self.session is not None and self.session is not session:
            raise ValidationError("runtime.session", "the already bound control session", session)
        if session.robot is not self.robot:
            raise ValidationError("runtime.session.robot", self.robot, session.robot)
        self._configure_session_period(session.period_s)
        super().bind_session(session)

    def restart(self, session: ControlSession) -> None:
        """Reset the simulation for a fresh single-use session, retaining the viewer."""
        if not self._opened or not self._used or self.session is None or self.session.active:
            raise ValidationError("runtime.restart", "an open runtime with a finished session", None)
        if session is self.session or session.robot is not self.robot:
            raise ValidationError("runtime.restart.session", "a fresh session for this robot", session)
        self._reset_for_replay()
        self.session = None
        self._used = False
        self.bind_session(session)

    def _reset_for_replay(self) -> None:
        raise NotImplementedError("This simulation backend does not support replay")

    def _configure_session_period(self, period_s: float) -> None:
        """Configure simulation substeps for the control interval."""

    @abstractmethod
    def advance_control_period(self) -> None:
        """Advance physics through one control interval."""

    @abstractmethod
    def _physics_time_s(self) -> float:
        """Read the physical simulation clock."""

    def _advance_interval(self) -> None:
        assert self.session is not None
        before = self._physics_time_s()
        if self._push.enabled and self.session.policy_tick and self._policy_simulation_time_s >= self._next_push_s:
            linear = self._push_rng.uniform(-np.asarray(self._push.linear), self._push.linear)
            angular = self._push_rng.uniform(-np.asarray(self._push.angular), self._push.angular)
            self._apply_root_velocity_push(linear, angular)
            logging.getLogger(__name__).info("Root push at %.2fs: dv=%s dw=%s", before, linear, angular)
            self._next_push_s = self._policy_simulation_time_s + self._push_rng.uniform(*self._push.interval_s)
        try:
            self.advance_control_period()
        finally:
            if self.session.policy_tick:
                self._policy_simulation_time_s += self._physics_time_s() - before

    def _sample_load(self):
        """Keep load dynamics fixed for the lifetime of this runtime."""
        cfg = self._randomization
        if self._load_draw is None:
            rng = self._physical_rng
            self._load_draw = (
                rng.uniform(*cfg.load_mass),
                rng.uniform(-np.asarray(cfg.load_com), cfg.load_com),
                rng.uniform(*cfg.load_inertia_factor),
                rng.uniform(*cfg.load_friction),
            )
        return self._load_draw

    def _initial_state_for_reset(self):
        from motrix_env_core.math.quaternion import from_euler, mul

        initial = self.config.initial_state
        cfg = self._randomization
        if initial is None or not cfg.enabled:
            return initial
        rng = self._physical_rng

        def noise(amplitude):
            return rng.uniform(-np.asarray(amplitude), amplitude)

        name = self.scene.objs.robot.resolved_base_link_name
        bodies = dict(initial.bodies)
        base = bodies[name]
        quat = mul(from_euler(*noise(cfg.base_rotation)), np.asarray(base.quaternion_xyzw))
        bodies[name] = replace(
            base,
            position=tuple(np.asarray(base.position) + noise(cfg.base_position)),
            quaternion_xyzw=tuple(quat / np.linalg.norm(quat)),
            linear_velocity=tuple(np.asarray(base.linear_velocity) + noise(cfg.base_linear_velocity)),
            angular_velocity=tuple(np.asarray(base.angular_velocity) + noise(cfg.base_angular_velocity)),
        )
        positions = dict(initial.joint_positions)
        for joint, low, high in zip(
            self.robot.spec.joint_names, self.robot.spec.position_lower, self.robot.spec.position_upper, strict=True
        ):
            if joint in positions:
                positions[joint] = float(
                    np.clip(positions[joint] + rng.uniform(-cfg.joint_position, cfg.joint_position), low, high)
                )
        return replace(initial, bodies=bodies, joint_positions=positions)

    def _apply_root_velocity_push(self, linear: np.ndarray, angular: np.ndarray) -> None:
        raise NotImplementedError("This simulation backend does not support root velocity pushes")

    def _run_session(self, scheduler: LoopScheduler) -> SimulationRolloutResult:
        self._next_push_s = self._push_rng.uniform(*self._push.interval_s)
        self._policy_simulation_time_s = 0.0
        result = super()._run_session(scheduler)
        return SimulationRolloutResult(
            **{item.name: getattr(result, item.name) for item in fields(RolloutResult)},
            policy_simulation_time_s=self._policy_simulation_time_s,
            real_time_factor=self._policy_simulation_time_s / result.wall_time_s if result.wall_time_s > 0 else 0.0,
        )

    def _check_running(self) -> None:
        self.check_viewer_running()

    def check_viewer_running(self) -> None:
        """Reject viewer shutdown; headless simulation hosts need no check."""


__all__ = ["DeploymentRuntime", "SimulationRuntime"]
