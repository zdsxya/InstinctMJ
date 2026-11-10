"""CLI for the independent multi-task teleoperation runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def _configure_gl(args: list[str]) -> None:
    """Choose GL before importing mjlab/MuJoCo (which cache the backend)."""
    viewer = "auto"
    for index, arg in enumerate(args):
        if arg == "--viewer" and index + 1 < len(args):
            viewer = args[index + 1]
        elif arg.startswith("--viewer="):
            viewer = arg.split("=", 1)[1]
    native = viewer == "native" or (
        viewer == "auto" and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    )
    if native:
        os.environ["MUJOCO_GL"] = "glfw"
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    _configure_gl(args)

    import tyro
    from .config import TeleopConfig, describe_config
    from .registry import build_env_config, get_task, list_tasks

    if not args or args[0] in ("--help", "-h", "--list-tasks"):
        print("usage: instinct-teleop <TASK> [OPTIONS]\n")
        print("Supported profiles:")
        for alias in list_tasks():
            print(f"  {alias:20s} {get_task(alias).task_id}")
        print("\nFull registered task IDs are also accepted.")
        print("Run instinct-teleop <TASK> --help for configuration options.")
        print("Run instinct-teleop <TASK> --print-config True to inspect resolved configuration.")
        return
    try:
        spec = get_task(args[0])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    # Match the existing project CLI: explicit True/False and Python tuples.
    flags = (tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff, tyro.conf.UsePythonSyntaxForLiteralCollections)
    cfg = tyro.cli(TeleopConfig, args=args[1:], default=spec.defaults(), prog=f"instinct-teleop {spec.alias}", config=flags)
    if cfg.task != spec.alias:
        raise SystemExit("Select the task as the first positional argument; do not override --task.")
    cfg.validate()
    if cfg.print_config:
        text = json.dumps(describe_config(cfg, build_env_config(cfg)), ensure_ascii=False, indent=2) + "\n"
        print(text, end="")
        if cfg.dump_config:
            path = Path(cfg.dump_config).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return
    from .runtime import run_teleop

    run_teleop(cfg)
