"""Browse retargeted G1 motions in Viser, without starting a training environment.

Run: .venv/bin/python scripts/umr/viewer.py --port 8080
"""

import argparse
import logging
import os
import time
from pathlib import Path
from queue import Empty, SimpleQueue

import mujoco
import numpy as np
import viser
from mjviser import ViserMujocoScene
from viewer_data import ROOT, MotionLibrary, Playback

LOG = logging.getLogger(__name__)


class MotionViewer:
    def __init__(self, library: MotionLibrary, host: str = "127.0.0.1", port: int = 8080):
        self.library = library
        self.model = library.model
        self.data = mujoco.MjData(self.model)
        self.playback = Playback()
        # Only the playback thread touches MuJoCo or applies UI state. Viser's
        # callbacks enqueue user events, never programmatic slider updates.
        self.commands = SimpleQueue()
        self.frame = 0
        self.in_place = False
        self.follow = True
        self.path_handles = []
        self.last_root = None
        self.name = ""
        self.server = viser.ViserServer(host=host, port=port, label="G1 · UMR Motion Viewer")
        self.server.gui.configure_theme(control_layout="collapsible", control_width="large", show_share_button=False)
        self.server.scene.set_up_direction("+z")
        self.scene = ViserMujocoScene(self.server, self.model, num_envs=1)
        # mjviser's tracking translates the entire scene, including height.
        # Here tracking moves only the camera, preserving the dataset's world frame.
        self.scene.camera_tracking_enabled = False
        self.scene.site_groups_visible = [False] * 6
        self.grid = self.server.scene.add_grid(
            "/ground",
            width=30,
            height=30,
            cell_size=0.25,
            section_size=1,
            plane="xy",
            plane_color=(225, 231, 239),
            plane_opacity=0.55,
        )
        self.axes = self.server.scene.add_frame("/world", axes_length=0.3, axes_radius=0.008)
        self._setup_gui()
        self._set_geometry("外观")
        self._load(self.motion_select.value)

        @self.server.on_client_connect
        def connected(client):
            self.commands.put(("connect", client.client_id))

    def _bind(self, handle, command, value=None, button=False):
        async def callback(event):
            # Capture the value before a subsequent message or server-side update
            # replaces it; threaded callbacks can otherwise read a newer value.
            if event.client_id is not None:
                self.commands.put((command, value if button else event.target.value))

        register = handle.on_click if button or isinstance(handle, viser.GuiButtonGroupHandle) else handle.on_update
        register(callback)

    def _setup_gui(self):
        self.server.gui.add_markdown("## G1 动作查看器\nUMR 重定向 · 运动学回放")
        show_all = not bool(self.library.selected)
        with self.server.gui.add_folder("动作库"):
            self.all_toggle = self.server.gui.add_checkbox("全部候选动作", initial_value=show_all)
            self.motion_select = self.server.gui.add_dropdown(
                "动作",
                options=self.library.names(show_all),
            )
            self._bind(self.all_toggle, "filter")
            self._bind(self.motion_select, "motion")
            self._bind(self.server.gui.add_button("上一个动作"), "next", -1, button=True)
            self._bind(self.server.gui.add_button("下一个动作"), "next", 1, button=True)
            self.info = self.server.gui.add_markdown("")
        self.playback_folder = self.server.gui.add_folder("播放")
        with self.playback_folder:
            self.play_button = self.server.gui.add_button("播放", icon=viser.Icon.PLAYER_PLAY, order=0)
            self._bind(self.play_button, "play", button=True)
            self._bind(self.server.gui.add_button("上一帧", order=1), "step", -1, button=True)
            self._bind(self.server.gui.add_button("下一帧", order=2), "step", 1, button=True)
            self.timeline = None
            self.status = self.server.gui.add_markdown("", order=4)
            speed = self.server.gui.add_dropdown(
                "倍速", options=("0.25×", "0.5×", "1×", "2×"), initial_value="1×", order=5
            )
            self._bind(speed, "speed")
            loop = self.server.gui.add_checkbox("循环播放", initial_value=True, order=6)
            self._bind(loop, "loop")
        with self.server.gui.add_folder("显示与视角"):
            geometry = self.server.gui.add_dropdown("机器人显示", options=("外观", "碰撞几何", "外观 + 碰撞"))
            self._bind(geometry, "geometry")
            self._bind(self.server.gui.add_checkbox("原地查看", initial_value=False), "in_place")
            self._bind(self.server.gui.add_checkbox("相机跟随", initial_value=True), "follow")
            self.root_path = self.server.gui.add_checkbox("根部轨迹", initial_value=False)
            self.feet_path = self.server.gui.add_checkbox("左右踝轨迹", initial_value=False)
            self._bind(self.root_path, "paths")
            self._bind(self.feet_path, "paths")
            self._bind(self.server.gui.add_checkbox("地面网格", initial_value=True), "grid")
            self._bind(self.server.gui.add_checkbox("世界坐标轴", initial_value=True), "axes")
            view = self.server.gui.add_button_group("视角", options=("正面", "侧面", "俯视", "重置"))
            self._bind(view, "camera")
            self.server.gui.add_markdown(
                "轨迹：根部为橙色，左踝为蓝色，右踝为红色。\n\n原地模式保留高度和朝向；踝轨迹不等同于足底接触轨迹。"
            )
        self.error = self.server.gui.add_markdown("")

    def _load(self, name):
        motion = self.library.load(name)  # Validate before replacing the current clip.
        # Publish the new range, pose and metadata together, so a new clip never
        # appears with the previous clip's timeline still active in the browser.
        with self.server.atomic():
            self._apply_motion(name, motion)
        self.server.flush()

    def _apply_motion(self, name, motion):
        self.motion = motion
        self.name = name
        self.frame = 0
        self.playback.position = 0
        self.playback.playing = False
        self.motion_select.value = name
        # Recreate the slider for each clip: in Viser 1.0.29 its numeric input
        # can retain stale bounds after changing the slider's max in place.
        if self.timeline is not None:
            self.timeline.remove()
        with self.playback_folder:
            self.timeline = self.server.gui.add_slider(
                "帧",
                min=0,
                max=max(1, len(motion.qpos) - 1),
                step=1,
                initial_value=0,
                disabled=len(motion.qpos) == 1,
                order=3,
            )
            self._bind(self.timeline, "seek")
        clip = self.library.metadata.get(Path(name).stem.removesuffix("_retargeted"), {})
        kind = {"walk": "步行", "turn": "转向", "stand": "站立"}.get(clip.get("kind"), "未标注")
        selected = (
            "已入选 AMP"
            if name in self.library.selected
            else ("未入选 AMP" if self.library.has_selection else "未提供 AMP 清单")
        )
        source = f"\n\n来源：`{clip['source']}`，{clip['start']}–{clip['end']} s" if clip else ""
        self.info.content = (
            f"**{kind} · {selected}**\n\n{len(motion.qpos)} 帧 · {motion.fps:g} Hz · {motion.duration:.2f} s{source}"
        )
        self.error.content = ""
        self.last_root = None
        self._rebuild_paths()
        self._render()
        self._camera("重置")

    def _set_geometry(self, mode):
        # The training G1 MJCF uses group 2 for visual meshes and group 3
        # for collision capsules. Preserve the source model and its materials.
        self.scene.geom_groups_visible = [False, False, mode != "碰撞几何", mode != "外观", False, False]
        # mjviser 0.0.14's public refresh only updates transforms. Apply the
        # group visibility flags as its own create_groups_gui() callback does.
        self.scene._sync_visibilities()
        self.scene.request_update()

    def _rebuild_paths(self):
        for handle in self.path_handles:
            handle.remove()
        self.path_handles.clear()
        paths = self.motion.trajectories(self.in_place)
        for i, (label, color) in enumerate(
            zip(("root", "left_ankle", "right_ankle"), ((245, 158, 11), (59, 130, 246), (239, 68, 68)))
        ):
            points = np.stack((paths[:-1, i], paths[1:, i]), axis=1).astype(np.float32)
            if len(points):
                self.path_handles.append(
                    self.server.scene.add_line_segments(
                        f"/trajectories/{label}",
                        points=points,
                        colors=color,
                        line_width=3,
                        visible=self.root_path.value if i == 0 else self.feet_path.value,
                    )
                )

    def _camera(self, preset, client_id=None):
        root = self.data.qpos[self.motion.root_address : self.motion.root_address + 3]
        target = np.array([root[0], root[1], 0.65])
        offsets = {"正面": (2, 0, 0.2), "侧面": (0, -2, 0.2), "俯视": (0, 0, 2.5), "重置": (1.7, -1.7, 0.85)}
        for cid, client in self.server.get_clients().items():
            if client_id is None or cid == client_id:
                with client.atomic():
                    client.camera.up_direction = (0, 1, 0) if preset == "俯视" else (0, 0, 1)
                    client.camera.position = target + offsets[preset]
                    client.camera.look_at = target

    def _render(self):
        self.data.qpos[:] = self.motion.pose(self.frame, self.in_place)
        mujoco.mj_forward(self.model, self.data)
        root = self.data.qpos[self.motion.root_address : self.motion.root_address + 3].copy()
        with self.server.atomic():
            self.scene.update_from_mjdata(self.data)
            if self.follow and self.last_root is not None:
                delta = root - self.last_root
                delta[2] = 0  # Keep camera height steady while the body moves vertically.
                if np.any(delta):
                    for client in self.server.get_clients().values():
                        with client.atomic():
                            client.camera.position = np.asarray(client.camera.position) + delta
                            client.camera.look_at = np.asarray(client.camera.look_at) + delta
            self.last_root = root
            self.timeline.value = self.frame
            self.status.content = (
                f"**{self.frame / self.motion.fps:.2f} / {self.motion.duration:.2f} s**"
                f" · 帧 {self.frame} / {len(self.motion.qpos) - 1}"
            )
            self.play_button.label = "暂停" if self.playback.playing else "播放"
            self.play_button.icon = viser.Icon.PLAYER_PAUSE if self.playback.playing else viser.Icon.PLAYER_PLAY

    def _command(self, command, value):
        if command == "motion":
            self._load(value)
        elif command == "filter":
            names = self.library.names(value)
            if not names:
                self.all_toggle.value = True
                raise ValueError("AMP 清单为空，请查看全部候选动作")
            self.motion_select.options = names
            self._load(self.name if self.name in names else names[0])
        elif command == "next":
            names = list(self.motion_select.options)
            self._load(names[(names.index(self.name) + value) % len(names)])
        elif command == "play":
            if not self.playback.playing and self.frame == len(self.motion.qpos) - 1:
                self.playback.position = 0
            self.playback.playing = not self.playback.playing
        elif command in ("seek", "step"):
            self.playback.playing = False
            target = value if command == "seek" else self.frame + value
            self.playback.position = int(np.clip(target, 0, len(self.motion.qpos) - 1))
        elif command == "speed":
            self.playback.speed = float(value.removesuffix("×"))
        elif command == "loop":
            self.playback.loop = value
        elif command == "geometry":
            self._set_geometry(value)
        elif command == "in_place":
            self.in_place = value
            self._rebuild_paths()
            self._render()
            self._camera("重置")
        elif command == "follow":
            self.follow = value
        elif command == "paths":
            for i, handle in enumerate(self.path_handles):
                handle.visible = self.root_path.value if i == 0 else self.feet_path.value
        elif command == "grid":
            self.grid.visible = value
        elif command == "axes":
            self.axes.visible = value
        elif command == "camera":
            self._camera(value)
        elif command == "connect":
            self._camera("重置", value)
        self.frame = self.playback.advance(0, self.motion.fps, len(self.motion.qpos))

    def run(self):
        previous = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                was_playing = self.playback.playing
                frame = self.playback.advance(now - previous, self.motion.fps, len(self.motion.qpos))
                dirty = frame != self.frame or was_playing != self.playback.playing
                self.frame = frame
                previous = now
                while True:
                    try:
                        command, value = self.commands.get_nowait()
                    except Empty:
                        break
                    try:
                        self._command(command, value)
                        self.error.content = ""
                    except (ValueError, KeyError, OSError) as exc:
                        LOG.exception("Viewer command failed")
                        self.error.content = f"**无法完成操作：** {exc}"
                        self.motion_select.value = self.name
                    dirty = True
                    previous = time.monotonic()  # Loading/seeking time is not playback time.
                if dirty:
                    self._render()
                time.sleep(1 / 120)
        finally:
            self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion-dir", type=Path, default=Path(os.environ.get("UMR_MOTION_DIR", ROOT / "outputs/umr/motions"))
    )
    parser.add_argument("--robot-xml", type=Path, help="Matching MJCF, for datasets moved to another machine")
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to allow remote browsers")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        library = MotionLibrary(args.motion_dir, args.robot_xml)
        viewer = MotionViewer(library, args.host, args.port)
        print(f"Loaded {len(library.files)} motions; {len(library.selected)} selected for AMP.", flush=True)
        viewer.run()
    except KeyboardInterrupt:
        pass
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Motion viewer: {exc}\n")


if __name__ == "__main__":
    main()
