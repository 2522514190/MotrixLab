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


@dataclass(frozen=True)
class SimulationRuntimeConfig:
    scene: SceneCfg
    physics: SimCfg = field(default_factory=lambda: SimCfg(dt=0.002, solver_iterations=100))
    render: bool = False
    realtime: bool | None = None
    # Deployment recipe: semantic RobotState fields mapped to model-local sensor names.
    sensor_bindings: SensorBindings = field(default_factory=SensorBindings)
    initial_state: SimulationInitialState | None = None

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
        expected = {"scene", "physics", "sensor_bindings"}
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
            render=render,
            realtime=realtime,
        )
