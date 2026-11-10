"""Externally controlled body-frame velocity with task-specific bounds.

Only the environment's command-manager update advances the command. This keeps
velocity observations, history buffers, and MILD's gait clock synchronized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from instinct_mj.teleop.config import ControlConfig


@dataclass(kw_only=True)
class ManualVelocityCommandCfg(CommandTermCfg):
    control: ControlConfig
    resampling_time_range: tuple[float, float] = (1e9, 1e9)

    def build(self, env) -> ManualVelocityCommand:
        return ManualVelocityCommand(self, env)


class ManualVelocityCommand(CommandTerm):
    """Accept targets in m/s and rad/s; apply acceleration limits per sim step.

    ``set_target`` updates the desired command, not observation history. The
    environment applies it during its next command update. ``stop`` clears both
    desired and applied commands immediately for pause/disconnect/reset.
    """

    def __init__(self, cfg: ManualVelocityCommandCfg, env):
        cfg.control.validate()
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 3), device=self.device)
        self._target = torch.zeros_like(self._command)
        ranges = (cfg.control.lin_vel_x, cfg.control.lin_vel_y, cfg.control.ang_vel_z)
        self._lower = torch.tensor([bounds[0] for bounds in ranges], device=self.device)
        self._upper = torch.tensor([bounds[1] for bounds in ranges], device=self.device)
        self._acceleration = torch.tensor(
            [cfg.control.linear_acceleration, cfg.control.linear_acceleration, cfg.control.yaw_acceleration],
            device=self.device,
        )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def target(self) -> torch.Tensor:
        return self._target

    def _velocity_tensor(self, velocity) -> torch.Tensor:
        values = torch.as_tensor(velocity, dtype=self._command.dtype, device=self.device)
        if values.shape not in ((3,), self._command.shape):
            raise ValueError(f"Expected velocity shape (3,) or {tuple(self._command.shape)}, got {tuple(values.shape)}")
        if not torch.isfinite(values).all():
            raise ValueError("Velocity commands must be finite")
        return values

    def set_target(self, velocity) -> None:
        """Set a physical target, clamped to the selected task's configured ranges."""
        values = self._velocity_tensor(velocity)
        self._target.copy_(torch.clamp(values, min=self._lower, max=self._upper))

    def set_normalized(self, velocity) -> None:
        """Map normalized stick axes to asymmetric physical ranges around zero."""
        values = self._velocity_tensor(velocity).clamp(-1.0, 1.0)
        self.set_target(torch.where(values >= 0, values * self._upper, values * -self._lower))

    def stop(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._target[ids] = 0.0
        self._command[ids] = 0.0

    def reset(self, env_ids=None) -> dict[str, float]:
        self.stop(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.command_counter[ids] = 0
        self.time_left[ids] = 0.0
        return {}

    def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        # An external source owns targets. Do not invoke automatic resampling,
        # heading control, or random standing masks from the training command.
        elapsed = torch.as_tensor(dt, dtype=self._command.dtype, device=self.device)
        if not torch.isfinite(elapsed).all() or torch.any(elapsed < 0):
            raise ValueError("Command update dt must be finite and nonnegative")
        if elapsed.ndim > 0:
            if env_ids is not None or elapsed.shape != (self.num_envs,):
                raise ValueError("Tensor dt requires shape (num_envs,) and no env_ids")
            elapsed = elapsed[:, None]
        ids = slice(None) if env_ids is None else env_ids
        maximum_change = elapsed * self._acceleration
        change = torch.clamp(self._target[ids] - self._command[ids], min=-maximum_change, max=maximum_change)
        self._command[ids] += change

    def on_viewer_pause(self, paused: bool) -> None:
        if paused:
            self.stop()

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        pass

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        # All state advancement is dt-aware and centralized in compute().
        pass
