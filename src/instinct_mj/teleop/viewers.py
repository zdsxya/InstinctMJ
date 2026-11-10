"""Render backends sharing the same teleoperation state machine."""

from __future__ import annotations

import html
import threading

from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.base import BaseViewer


class TeleopViewerMixin:
    # Interaction uses a real-time target; actual rate remains visible in the UI.
    SPEED_MULTIPLIERS = [1.0]

    def poll_input(self):
        session = self.env
        state = session.source.poll()
        if state.error and state.error != session.input_error:
            print(f"[teleop] {state.error}; waiting for input device.")
        session.input_error = state.error
        if state.connected != session.connected:
            print("[teleop] Input connected." if state.connected else "[teleop] Input disconnected; paused.")
        session.connected = state.connected
        if state.quit:
            session.quit_requested = True
            self.pause()
            return
        if not state.connected:
            self.pause()
            return
        if state.reset:
            self.reset_environment()
        elif state.pause:
            self.toggle_pause()
        if session.needs_neutral:
            session.needs_neutral = any(abs(value) > 1e-6 for value in state.velocity)
        if self._is_paused or session.needs_neutral or session.needs_reset:
            session.command.stop()
        else:
            session.command.set_normalized(state.velocity)

    def tick(self):
        self.poll_input()
        return super().tick()

    def pause(self):
        super().pause()
        self.env.command.stop()
        self.env.needs_neutral = True

    def resume(self):
        if self.env.needs_reset or not self.env.connected:
            return
        super().resume()

    def reset_environment(self):
        super().reset_environment()
        self.pause()

    def _execute_step(self):
        if self._max_steps is not None and self.env.total_steps >= self._max_steps:
            return False
        if self.env.needs_reset or not self.env.connected or self.env.quit_requested:
            return False
        success = super()._execute_step()
        if self.env.needs_reset:
            print("[teleop] Episode terminated; paused. Reset to continue.")
            self.pause()
            return False  # Stop the current frame's catch-up steps too.
        return success


class TeleopNativeViewer(TeleopViewerMixin, NativeMujocoViewer):
    def setup(self):
        import mujoco.viewer

        existing = set(threading.enumerate())
        self._render_threads = []
        try:
            super().setup()
            if self.cfg.origin_type == self.cfg.OriginType.WORLD:
                self._set_camera_world()
        finally:
            self._render_threads = [
                thread for thread in threading.enumerate()
                if thread not in existing and getattr(thread, "_target", None) is mujoco.viewer._launch_internal
            ]

    def close(self):
        super().close()
        for thread in getattr(self, "_render_threads", ()):
            thread.join()
        self._render_threads = []

    def _set_status_overlay(self, viewer):
        import mujoco

        status = self.get_status()
        rows = [
            ("Teleop", "PAUSED" if status.paused else "RUNNING"),
            ("Steps", str(self.env.total_steps)),
            ("Actual RT", f"{status.actual_realtime:.2f}x"),
            *self.env.status_lines(),
            ("Keyboard", "Space: pause | Enter: reset"),
        ]
        if status.last_error:
            rows.append(("Error", status.last_error.strip().splitlines()[-1]))
        viewer.set_texts((
            mujoco.mjtFontScale.mjFONTSCALE_150.value,
            mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
            "\n".join(row[0] for row in rows),
            "\n".join(row[1] for row in rows),
        ))


class TeleopViserViewer(TeleopViewerMixin, ViserPlayViewer):
    def _update_status_display(self):
        super()._update_status_display()
        rows = "<br/>".join(f"<b>{html.escape(k)}:</b> {html.escape(v)}" for k, v in self.env.status_lines())
        self._status_html.content += f"<div style='padding:8px'>{rows}</div>"

    def close(self):
        # Base setup can fail before creating its worker pool.
        if hasattr(self, "_threadpool"):
            super().close()
        elif not self._external_server:
            self._server.stop()


class TeleopHeadlessViewer(TeleopViewerMixin, BaseViewer):
    def setup(self):
        pass

    def sync_env_to_viewer(self):
        pass

    def sync_viewer_to_env(self):
        pass

    def is_running(self):
        return True

    def close(self):
        pass


def make_viewer(backend, session, policy, cfg):
    classes = {"native": TeleopNativeViewer, "viser": TeleopViserViewer, "none": TeleopHeadlessViewer}
    viewer = classes[backend](session, policy)
    viewer._is_paused = cfg.start_paused
    viewer._max_steps = cfg.max_steps
    return viewer
