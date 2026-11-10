"""Explicit teleop task registration, layered on the project's task registry."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from instinct_mj.teleop.config import TeleopConfig


@dataclass(frozen=True)
class TaskSpec:
    alias: str
    task_id: str
    config_module: str
    adapter_entry_point: str
    base_config_module: str

    def defaults(self) -> TeleopConfig:
        cfg = importlib.import_module(self.config_module).default_config()
        cfg.task = self.alias
        return cfg

    def adapter(self):
        module, name = self.adapter_entry_point.split(":")
        return getattr(importlib.import_module(module), name)(self.task_id)

    @staticmethod
    def _path(module: str) -> str:
        module_path = importlib.util.find_spec(module)
        if module_path is None or module_path.origin is None:
            raise ImportError(f"Cannot locate configuration module {module}")
        return str(Path(module_path.origin).resolve())

    @property
    def config_path(self) -> str:
        return self._path(self.config_module)

    @property
    def adapter_path(self) -> str:
        return self._path(self.adapter_entry_point.split(":")[0])

    @property
    def base_config_path(self) -> str:
        return self._path(self.base_config_module)


_TASKS: dict[str, TaskSpec] = {}


def register_task(spec: TaskSpec) -> None:
    """Register an explicitly supported task; task IDs are never guessed by substring."""
    if spec.alias in _TASKS or any(item.task_id == spec.task_id for item in _TASKS.values()):
        raise ValueError(f"Teleop task already registered: {spec.alias}")
    _TASKS[spec.alias] = spec


def list_tasks() -> list[str]:
    return sorted(_TASKS)


def get_task(name: str) -> TaskSpec:
    if name in _TASKS:
        return _TASKS[name]
    for task in _TASKS.values():
        if name in (task.task_id, task.task_id.removesuffix("-v0") + "-Play-v0"):
            return task
    raise ValueError(f"Unsupported teleop task {name!r}. Choose from: {', '.join(list_tasks())}")


def build_env_config(cfg: TeleopConfig):
    """Return an isolated interactive environment configuration for this profile."""
    cfg.validate()
    return get_task(cfg.task).adapter().prepare(cfg)


register_task(TaskSpec(
    alias="mild_walk",
    task_id="Instinct-Locomotion-Mild-Walk-G1-v0",
    config_module="instinct_mj.teleop.tasks.mild_walk.config",
    adapter_entry_point="instinct_mj.teleop.tasks.mild_walk.adapter:MildWalkAdapter",
    base_config_module="instinct_mj.tasks.locomotion.config.g1.mild_walk_env_cfg",
))
for _alias, _task_id in (
    ("parkour_amp", "Instinct-Parkour-Target-Amp-G1-v0"),
    ("parkour_supermimic", "Instinct-Parkour-Target-Amp-SuperMimic-G1-v0"),
):
    register_task(TaskSpec(
        alias=_alias,
        task_id=_task_id,
        config_module="instinct_mj.teleop.tasks.parkour.config",
        adapter_entry_point="instinct_mj.teleop.tasks.parkour.adapter:ParkourAdapter",
        base_config_module="instinct_mj.tasks.parkour.config.g1.g1_parkour_target_amp_cfg",
    ))
