"""Nonblocking Linux joystick input; no SDL or window focus is required.

The gamepad must be connected to the computer running the Python process, even
when the simulation is viewed through a browser on another computer.
"""

from __future__ import annotations

import math
import os
import struct
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class GamepadConfig:
    """Linux js axis/button indices; defaults follow common Xbox mappings.

    Forward, left, and counterclockwise turn are positive. Use ``None`` to
    disable an axis or button. Axis numbering differs between device drivers.
    """

    device: str = "/dev/input/js0"
    forward_axis: int | None = 1
    lateral_axis: int | None = 0
    yaw_axis: int | None = 3
    forward_sign: float = -1.0
    lateral_sign: float = -1.0
    yaw_sign: float = -1.0
    deadzone: float = 0.1
    pause_button: int | None = 7
    reset_button: int | None = 0
    quit_button: int | None = 6
    reconnect_interval: float = 1.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.device:
            raise ValueError("gamepad.device must not be empty")
        for name in ("forward_axis", "lateral_axis", "yaw_axis", "pause_button", "reset_button", "quit_button"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ValueError(f"gamepad.{name} must be a nonnegative integer or None")
        for name in ("forward_sign", "lateral_sign", "yaw_sign"):
            if getattr(self, name) not in (-1.0, 1.0):
                raise ValueError(f"gamepad.{name} must be -1 or 1")
        if not math.isfinite(self.deadzone) or not 0 <= self.deadzone < 1:
            raise ValueError("gamepad.deadzone must be finite and in [0, 1)")
        if not math.isfinite(self.reconnect_interval) or self.reconnect_interval <= 0:
            raise ValueError("gamepad.reconnect_interval must be finite and positive")
        buttons = [value for value in (self.pause_button, self.reset_button, self.quit_button) if value is not None]
        if len(buttons) != len(set(buttons)):
            raise ValueError("gamepad pause/reset/quit buttons must be distinct")


@dataclass(frozen=True)
class InputState:
    """One input poll: normalized body-frame velocity and rising-edge actions."""

    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    connected: bool = True
    pause: bool = False
    reset: bool = False
    quit: bool = False
    error: str | None = None


def apply_deadzone(value: float, deadzone: float) -> float:
    """Remove the deadzone and continuously remap the remaining range to [-1, 1]."""
    value = max(-1.0, min(1.0, value))
    if abs(value) <= deadzone:
        return 0.0
    return math.copysign((abs(value) - deadzone) / (1.0 - deadzone), value)


class LinuxGamepad:
    """Read Linux ``js_event`` packets without blocking the simulation loop.

    A stationary stick does not generate events, so lack of new packets is not
    treated as a disconnect. EOF/read errors immediately clear all stick state.
    Reconnection is retried on subsequent polls. Runtime decides when to resume.
    """

    _EVENT = struct.Struct("<IhBB")
    _BUTTON = 0x01
    _AXIS = 0x02
    _INIT = 0x80

    def __init__(self, cfg: GamepadConfig):
        if not sys.platform.startswith("linux"):
            raise RuntimeError("The gamepad backend requires Linux /dev/input/js*. Use --input zero for a smoke run.")
        self.cfg = cfg
        self._fd: int | None = None
        self._next_open = 0.0
        self._axes: dict[int, float] = {}
        self._buttons: dict[int, bool] = {}
        self._buffer = bytearray()
        self._closed = False
        self.error: str | None = None

    def _disconnect(self, reason: str) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
        self._fd = None
        self._axes.clear()
        self._buttons.clear()
        self._buffer.clear()
        self.error = reason
        self._next_open = time.monotonic() + self.cfg.reconnect_interval

    def _connect(self) -> bool:
        if self._closed or time.monotonic() < self._next_open:
            return False
        try:
            self._fd = os.open(self.cfg.device, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        except OSError as exc:
            self._disconnect(f"{self.cfg.device}: {exc.strerror}")
            return False
        self.error = None
        return True

    def poll(self) -> InputState:
        if self._fd is None and not self._connect():
            return InputState(connected=False, error=self.error)
        pressed: set[int] = set()
        # Bound the work per frame even when a device continuously emits events.
        for _ in range(32):
            try:
                chunk = os.read(self._fd, self._EVENT.size * 64)
            except BlockingIOError:
                break
            except OSError as exc:
                self._disconnect(f"{self.cfg.device}: {exc.strerror}")
                return InputState(connected=False, error=self.error)
            if not chunk:
                self._disconnect(f"{self.cfg.device}: disconnected")
                return InputState(connected=False, error=self.error)
            self._buffer.extend(chunk)
            while len(self._buffer) >= self._EVENT.size:
                _, value, event_type, number = self._EVENT.unpack_from(self._buffer)
                del self._buffer[: self._EVENT.size]
                initial = bool(event_type & self._INIT)
                event_type &= ~self._INIT
                if event_type == self._AXIS:
                    self._axes[number] = max(-1.0, min(1.0, value / 32767.0))
                elif event_type == self._BUTTON:
                    held = value != 0
                    if held and not self._buttons.get(number, False) and not initial:
                        pressed.add(number)
                    self._buttons[number] = held

        def axis(index: int | None, sign: float) -> float:
            return apply_deadzone(self._axes.get(index, 0.0) * sign, self.cfg.deadzone)

        return InputState(
            velocity=(
                axis(self.cfg.forward_axis, self.cfg.forward_sign),
                axis(self.cfg.lateral_axis, self.cfg.lateral_sign),
                axis(self.cfg.yaw_axis, self.cfg.yaw_sign),
            ),
            connected=True,
            pause=self.cfg.pause_button in pressed,
            reset=self.cfg.reset_button in pressed,
            quit=self.cfg.quit_button in pressed,
        )

    def close(self) -> None:
        self._closed = True
        self._disconnect("gamepad closed")


class ZeroInput:
    """Always connected zero input for headless verification without a gamepad."""

    def poll(self) -> InputState:
        return InputState()

    def close(self) -> None:
        pass
