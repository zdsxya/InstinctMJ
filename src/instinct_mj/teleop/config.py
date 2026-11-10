"""Shared teleoperation settings; task defaults live under ``teleop/tasks``."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal

from tyro.conf import Suppress

from instinct_mj.teleop.inputs.gamepad import GamepadConfig


@dataclass
class ControlConfig:
    """Body-frame command limits (m/s, rad/s) and slew rates (m/s², rad/s²)."""

    lin_vel_x: tuple[float, float] = (-0.5, 1.0)
    lin_vel_y: tuple[float, float] = (0.0, 0.0)
    ang_vel_z: tuple[float, float] = (-1.0, 1.0)
    linear_acceleration: float = 1.0
    yaw_acceleration: float = 2.0

    def validate(self) -> None:
        for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds):
                raise ValueError(f"control.{name} must contain two finite numbers")
            if not bounds[0] <= 0.0 <= bounds[1]:
                raise ValueError(f"control.{name} must have min <= 0 <= max")
        for name in ("linear_acceleration", "yaw_acceleration"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"control.{name} must be finite and positive")


@dataclass
class SceneConfig:
    """Task terrain selection and deterministic reset offsets."""

    terrain: str = "task"
    """MILD: task/plane. Parkour: task or a name from its sub_terrains dictionary."""
    terrain_rows: int = 1
    terrain_cols: int = 1
    terrain_difficulty: float = 0.2
    spawn_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset from the task's terrain-aware default spawn (meters)."""
    spawn_yaw: float = 0.0
    debug_vis: bool = False

    def validate(self) -> None:
        if self.terrain_rows < 1 or self.terrain_cols < 1:
            raise ValueError("scene.terrain_rows and scene.terrain_cols must be positive")
        if not math.isfinite(self.terrain_difficulty) or not 0 <= self.terrain_difficulty <= 1:
            raise ValueError("scene.terrain_difficulty must be in [0, 1]")
        if len(self.spawn_offset) != 3 or not all(math.isfinite(v) for v in self.spawn_offset):
            raise ValueError("scene.spawn_offset must contain three finite numbers")
        if not math.isfinite(self.spawn_yaw):
            raise ValueError("scene.spawn_yaw must be finite")


@dataclass
class CameraConfig:
    follow: bool = True
    distance: float = 3.5
    elevation: float = -15.0
    azimuth: float = 135.0
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.8)

    def validate(self) -> None:
        if not math.isfinite(self.distance) or self.distance <= 0:
            raise ValueError("camera.distance must be finite and positive")
        if len(self.lookat) != 3 or not all(math.isfinite(v) for v in self.lookat):
            raise ValueError("camera.lookat must contain three finite numbers")
        if not all(math.isfinite(v) for v in (self.elevation, self.azimuth)):
            raise ValueError("Camera angles must be finite")


@dataclass
class TeleopConfig:
    task: Suppress[str] = "mild_walk"
    checkpoint_file: str | None = None
    agent: Literal["trained", "zero"] = "trained"
    device: str | None = None
    viewer: Literal["auto", "native", "viser", "none"] = "auto"
    input: Literal["gamepad", "zero"] = "gamepad"
    max_steps: int | None = None
    start_paused: bool = True
    seed: int = 0
    print_config: bool = False
    """Print resolved settings and exit without creating a simulation."""
    dump_config: str | None = None
    """Optional JSON path for the resolved settings and their source files."""
    control: ControlConfig = field(default_factory=ControlConfig)
    gamepad: GamepadConfig = field(default_factory=GamepadConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)

    def validate(self) -> None:
        self.control.validate()
        self.gamepad.validate()
        self.scene.validate()
        self.camera.validate()
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


def describe_config(cfg: TeleopConfig, env_cfg=None) -> dict:
    """JSON-ready settings plus a short map to the files that define them."""
    from instinct_mj.teleop.registry import get_task

    task = get_task(cfg.task)
    result = {
        "task_id": task.task_id,
        "settings": asdict(cfg),
        "sources": {
            "shared_defaults": __file__,
            "task_defaults": task.config_path,
            "task_adapter": task.adapter_path,
            "base_task": task.base_config_path,
            "precedence": "base task -> shared teleop defaults -> task teleop defaults -> CLI",
        },
    }
    if env_cfg is not None:
        result["environment"] = {
            "num_envs": env_cfg.scene.num_envs,
            "physics_dt": env_cfg.sim.mujoco.timestep,
            "decimation": env_cfg.decimation,
            "policy_dt": env_cfg.sim.mujoco.timestep * env_cfg.decimation,
            "auto_reset": env_cfg.auto_reset,
            "observation_groups": list(env_cfg.observations),
            "sensors": [sensor.name for sensor in env_cfg.scene.sensors],
            "terminations": list(env_cfg.terminations),
        }
    return result
