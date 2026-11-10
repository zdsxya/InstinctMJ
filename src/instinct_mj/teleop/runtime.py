"""Shared teleoperation session; task physics stay in the registered environment."""

from __future__ import annotations

import os
from pathlib import Path
import time


class TeleopSession:
    """Viewer environment protocol plus input/termination state.

    Observations are cached from reset/step. Reading a gamepad or rendering never
    advances observation history. A new target enters the next command-manager
    update, followed by the corresponding policy observations (one control step).
    """

    def __init__(self, vec_env, command, source):
        self.vec_env = vec_env
        self.command = command
        self.source = source
        self.num_envs = vec_env.num_envs
        self.connected = False
        self.input_error = None
        self.needs_neutral = True
        self.needs_reset = False
        self.quit_requested = False
        self.total_steps = 0
        self.observations, _ = vec_env.get_observations()

    @property
    def unwrapped(self):
        return self.vec_env.unwrapped

    @property
    def cfg(self):
        return self.vec_env.cfg

    @property
    def device(self):
        return self.vec_env.device

    def get_observations(self):
        return self.observations

    def step(self, actions):
        if self.needs_reset:
            raise RuntimeError("Robot terminated; reset before continuing.")
        result = self.vec_env.step(actions)
        self.observations = result[0]
        self.total_steps += 1
        self.needs_reset = bool(result[2].any())
        if self.needs_reset:
            self.command.stop()
        return result

    def reset(self):
        self.command.stop()
        result = self.vec_env.reset()
        self.observations = result[0]
        self.needs_reset = False
        self.needs_neutral = True
        return result

    def close(self):
        self.vec_env.close()

    def status_lines(self):
        target = self.command.target[0].detach().cpu().tolist()
        command = self.command.command[0].detach().cpu().tolist()
        robot = self.unwrapped.scene["robot"]
        linear = robot.data.root_link_lin_vel_b[0].detach().cpu().tolist()
        angular = robot.data.root_link_ang_vel_b[0].detach().cpu().tolist()
        fmt = lambda values: ", ".join(f"{v:+.2f}" for v in values)
        state = "connected" if self.connected else "disconnected"
        if self.needs_reset:
            state += " / reset required"
        elif self.needs_neutral:
            state += " / center sticks"
        if self.input_error:
            state += f" / {self.input_error}"
        return [
            ("Input", state),
            ("Target vx,vy,wz", fmt(target)),
            ("Command vx,vy,wz", fmt(command)),
            ("Actual vx,vy,wz", fmt([linear[0], linear[1], angular[2]])),
        ]


def resolve_viewer(viewer: str) -> str:
    if viewer != "auto":
        return viewer
    return "native" if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") else "viser"


def run_teleop(cfg) -> None:
    """Build the selected task once and run the common interactive loop."""
    import torch
    from mjlab.utils.torch import configure_torch_backends

    from instinct_mj.tasks.registry import load_env_cls, load_instinct_rl_cfg, load_vecenv_cls
    from .inputs.gamepad import LinuxGamepad, ZeroInput
    from .policies.loader import load_policy, ZeroPolicy
    from .registry import get_task
    from .viewers import make_viewer

    spec = get_task(cfg.task)
    cfg.validate()
    backend = resolve_viewer(cfg.viewer)
    if backend == "none" and cfg.input == "zero":
        if cfg.start_paused:
            raise ValueError("Zero-input headless runs need --start-paused False.")
        if cfg.max_steps is None:
            raise ValueError("Zero-input headless runs need --max-steps to bound the rollout.")
    if cfg.agent == "trained":
        if not cfg.checkpoint_file:
            raise ValueError("Provide --checkpoint-file for a trained policy (or --agent zero for a smoke test).")
        if not Path(cfg.checkpoint_file).expanduser().is_file():
            raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_file}")

    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter = spec.adapter()
    env_cfg = adapter.prepare(cfg)
    if cfg.dump_config:
        from .config import describe_config
        import json

        output = Path(cfg.dump_config).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        snapshot = describe_config(cfg, env_cfg)
        snapshot["runtime"] = {"device": device, "viewer": backend}
        output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[teleop] Configuration saved to {output.resolve()}")
    source = LinuxGamepad(cfg.gamepad) if cfg.input == "gamepad" else ZeroInput()
    env = None
    viewer = None
    try:
        env = load_env_cls(spec.task_id)(cfg=env_cfg, device=device)
        agent_cfg = load_instinct_rl_cfg(spec.task_id)
        vec_cls = load_vecenv_cls(spec.task_id)
        vec_env = vec_cls(
            env, policy_group=agent_cfg.policy_observation_group,
            critic_group=agent_cfg.critic_observation_group,
        )
        command = adapter.bind_command(env)
        policy = (
            ZeroPolicy(vec_env.num_actions, device)
            if cfg.agent == "zero"
            else load_policy(spec.task_id, cfg.checkpoint_file, vec_env, device)
        )
        session = TeleopSession(vec_env, command, source)
        viewer = make_viewer(backend, session, policy, cfg)
        print(f"[teleop] {spec.task_id} | {device} | {backend} | policy {1 / env.step_dt:g} Hz")
        print("[teleop] Center sticks to enable control. Pause/resume and reset are available on the gamepad and viewer.")
        viewer.setup()
        now = time.perf_counter()
        viewer._stats_last_time = viewer._last_tick_time = now
        # Poll before deciding whether a requested unpaused start is possible.
        viewer.poll_input()
        if cfg.start_paused:
            viewer.pause()
        else:
            viewer.resume()
        while viewer.is_running() and not session.quit_requested:
            if cfg.max_steps is not None and session.total_steps >= cfg.max_steps:
                break
            if not viewer.tick():
                time.sleep(0.001)
            viewer._update_stats()
            if backend == "none" and (session.needs_reset or viewer._last_error):
                if viewer._last_error:
                    raise RuntimeError(viewer._last_error)
                break
        print(f"[teleop] Finished after {session.total_steps} steps" + (" (reset required)." if session.needs_reset else "."))
    except KeyboardInterrupt:
        print("\n[teleop] Stopped.")
    finally:
        try:
            if viewer is not None:
                viewer.close()
        finally:
            try:
                source.close()
            finally:
                if env is not None:
                    env.close()
