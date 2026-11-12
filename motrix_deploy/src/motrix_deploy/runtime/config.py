# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Simulator-independent deployment scene and state-source configuration."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from motrix_deploy.errors import ValidationError
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.config.sim import SimCfg


@dataclass
class SensorBindings:
    """Semantic robot state fields bound to exact model-local sensor names.

    Also serves as the structured Hydra schema inside ``SimRuntimeCfg``:
    mutable so composed recipes can override sensor names.
    """

    base_angular_velocity: str | None = None
    base_linear_acceleration: str | None = None
    base_linear_velocity: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "base_angular_velocity",
            "base_linear_acceleration",
            "base_linear_velocity",
        ):
            name = getattr(self, field_name)
            if name is None:
                continue
            if not isinstance(name, str):
                raise TypeError(f"{field_name} must be a string or None")
            if not name.strip():
                raise ValueError(f"{field_name} must be a non-empty sensor name")


@dataclass(frozen=True)
class FloatingBodyState:
    """Initial world pose (xyzw) and world linear/angular velocities."""

    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name, size in (("position", 3), ("quaternion_xyzw", 4), ("linear_velocity", 3), ("angular_velocity", 3)):
            values = getattr(self, name)
            if len(values) != size or not all(math.isfinite(value) for value in values):
                raise ValueError(f"initial_state.{name} must contain {size} finite values")
        if not math.isclose(sum(value * value for value in self.quaternion_xyzw), 1.0, abs_tol=1e-5):
            raise ValueError("initial_state.quaternion_xyzw must be a unit quaternion")


@dataclass(frozen=True)
class SimulationInitialState:
    """Task-owned initial state, with joints and floating bodies addressed by name."""

    bodies: Mapping[str, FloatingBodyState]
    joint_positions: Mapping[str, float]
    joint_velocities: Mapping[str, float]

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for values in (self.joint_positions, self.joint_velocities)
            for value in values.values()
        ):
            raise ValueError("initial_state joint values must be finite")


@dataclass
class VelocityPushCfg:
    """Independent world-axis velocity increments, sampled on policy time."""

    enabled: bool = False
    seed: int = 1
    interval_s: tuple[float, float] = (1.0, 3.0)
    linear: tuple[float, float, float] = (0.3, 0.3, 0.3)
    angular: tuple[float, float, float] = (0.78, 0.78, 0.78)

    def __post_init__(self) -> None:
        low, high = self.interval_s
        if not 0 < low <= high or not math.isfinite(high):
            raise ValueError("push.interval_s requires positive finite ordered bounds")
        for amplitude in (self.linear, self.angular):
            if len(amplitude) != 3 or any(not math.isfinite(v) or v < 0 for v in amplitude):
                raise ValueError("push amplitudes require three finite nonnegative values")


@dataclass
class SimulationRandomizationCfg:
    """Named load/body perturbations for simulation-only robustness evaluation."""

    enabled: bool = False
    seed: int = 1
    load_body: str | None = None
    load_geom: str | None = None
    com_body: str | None = None
    load_mass: tuple[float, float] = (0.1, 0.5)
    load_com: tuple[float, float, float] = (0.02, 0.02, 0.02)
    load_inertia_factor: tuple[float, float] = (0.8, 1.2)
    load_friction: tuple[float, float] = (0.5, 0.8)
    body_com: tuple[float, float, float] = (0.025, 0.05, 0.075)
    base_position: tuple[float, float, float] = (0.05, 0.05, 0.01)
    base_rotation: tuple[float, float, float] = (0.1, 0.1, 0.2)
    base_linear_velocity: tuple[float, float, float] = (0.5, 0.5, 0.2)
    base_angular_velocity: tuple[float, float, float] = (0.52, 0.52, 0.78)
    joint_position: float = 0.1
    observation_noise: bool = True

    def __post_init__(self) -> None:
        for name in ("load_mass", "load_inertia_factor", "load_friction"):
            low, high = getattr(self, name)
            if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
                raise ValueError(f"randomization.{name} requires finite ordered nonnegative bounds")
        if self.load_mass[0] <= 0 or self.load_inertia_factor[0] <= 0:
            raise ValueError("Randomized mass and inertia must be positive")
        for name in (
            "load_com",
            "body_com",
            "base_position",
            "base_rotation",
            "base_linear_velocity",
            "base_angular_velocity",
        ):
            value = getattr(self, name)
            if len(value) != 3 or any(not math.isfinite(v) or v < 0 for v in value):
                raise ValueError(f"randomization.{name} requires three nonnegative finite amplitudes")
        if not math.isfinite(self.joint_position) or self.joint_position < 0:
            raise ValueError("randomization.joint_position must be finite and nonnegative")


@dataclass(frozen=True)
class SimulationRuntimeConfig:
    scene: SceneCfg
    physics: SimCfg = field(default_factory=lambda: SimCfg(dt=0.002, solver_iterations=100))
    render: bool = False
    realtime: bool | None = None
    # Deployment recipe: semantic RobotState fields mapped to model-local sensor names.
    sensor_bindings: SensorBindings = field(default_factory=SensorBindings)
    initial_state: SimulationInitialState | None = None

    push: VelocityPushCfg = field(default_factory=VelocityPushCfg)
    randomization: SimulationRandomizationCfg = field(default_factory=SimulationRandomizationCfg)

    def __post_init__(self) -> None:
        self.physics.validate()

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, render: bool = False, realtime: bool | None = None
    ) -> "SimulationRuntimeConfig":
        """Decode an assembled simulation recipe at the application boundary."""
        values = dict(value)
        values.pop("name", None)
        values.setdefault("sensor_bindings", {})
        values.setdefault("push", {})
        values.setdefault("randomization", {})
        expected = {"scene", "physics", "sensor_bindings", "push", "randomization"}
        if set(values) != expected:
            raise ValidationError(
                "runtime",
                f"fields {sorted(expected)}",
                f"missing={sorted(expected - set(values))}, unknown={sorted(set(values) - expected)}",
            )
        if not isinstance(values["scene"], SceneCfg):
            raise ValidationError("runtime.scene", "a complete SceneCfg", values["scene"])
        missing = {"dt", "solver_iterations"} - set(values["physics"])
        if missing:
            raise ValidationError("runtime.physics", "dt and solver_iterations", f"missing={sorted(missing)}")
        return cls(
            scene=values["scene"],
            physics=SimCfg(**values["physics"]),
            sensor_bindings=SensorBindings(**values["sensor_bindings"]),
            push=VelocityPushCfg(**values["push"]),
            randomization=SimulationRandomizationCfg(**values["randomization"]),
            render=render,
            realtime=realtime,
        )
