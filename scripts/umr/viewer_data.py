"""Validated motion loading and playback timing, independent of the web UI."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = ROOT / "src/instinct_mj/assets/resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml"


class MotionLibrary:
    def __init__(self, directory: Path, robot_xml: Path | None = None):
        self.directory = directory.expanduser().resolve()
        manifest_path = self.directory.parent / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        self.metadata = {c["id"]: c for c in self.manifest.get("clips", [])}
        self.files = {p.name: p for p in sorted(self.directory.glob("*_retarget*.npz"))}
        if not self.files:
            raise ValueError(f"No retargeted motion files in {self.directory}")
        selection = self.directory / "motions.yaml"
        self.has_selection = selection.exists()
        self.selected = (
            list((yaml.safe_load(selection.read_text()) or {}).get("selected_files", [])) if self.has_selection else []
        )
        missing = set(self.selected) - self.files.keys()
        if missing:
            raise ValueError(f"motions.yaml references missing files: {sorted(missing)}")
        recorded = Path(self.manifest.get("robot_xml", DEFAULT_XML)).expanduser()
        if not recorded.is_absolute():
            recorded = manifest_path.parent / recorded
        self.robot_xml = (robot_xml.expanduser() if robot_xml else recorded).resolve()
        if not self.robot_xml.exists() and robot_xml is None:
            self.robot_xml = DEFAULT_XML
        expected_hash = self.manifest.get("robot_sha256")
        if expected_hash and hashlib.sha256(self.robot_xml.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Robot MJCF differs from the dataset manifest; use the matching --robot-xml")
        self.model = mujoco.MjModel.from_xml_path(str(self.robot_xml))
        self.cache: dict[str, Motion] = {}

    def names(self, show_all: bool) -> list[str]:
        return list(self.files) if show_all or not self.has_selection else self.selected.copy()

    def load(self, name: str) -> "Motion":
        if name not in self.cache:
            self.cache[name] = load_motion(self.files[name], self.model)
        return self.cache[name]


@dataclass
class Motion:
    qpos: np.ndarray
    fps: float
    root_address: int
    paths: np.ndarray  # frame, (root, left ankle, right ankle), xyz

    @property
    def duration(self) -> float:
        return (len(self.qpos) - 1) / self.fps

    def pose(self, frame: int, in_place: bool = False) -> np.ndarray:
        result = self.qpos[frame].copy()
        if in_place:
            result[self.root_address : self.root_address + 2] = 0
        return result

    def trajectories(self, in_place: bool = False) -> np.ndarray:
        result = self.paths.copy()
        if in_place:
            result[:, :, :2] -= self.paths[:, :1, :2]
        return result


def load_motion(path: Path, model: mujoco.MjModel) -> Motion:
    """Map named joints to MJCF addresses; file quaternions are MuJoCo wxyz."""
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    hinges = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    if len(free) != 1 or len(hinges) + 1 != model.njnt:
        raise ValueError("Expected one floating root and scalar hinge joints")
    root = int(model.jnt_qposadr[free[0]])
    with np.load(path, allow_pickle=False) as arrays:
        fps = float(arrays["framerate"].item())
        names = arrays["joint_names"].tolist()
        joint_pos = np.asarray(arrays["joint_pos"], dtype=float)
        pos = np.asarray(arrays["base_pos_w"], dtype=float)
        quat = np.asarray(arrays["base_quat_w"], dtype=float)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid frame rate in {path.name}")
    expected = [model.joint(int(j)).name for j in hinges]
    if len(set(names)) != len(names) or set(names) != set(expected):
        raise ValueError(f"Joint names do not match MJCF in {path.name}")
    n = len(pos)
    if n == 0 or pos.shape != (n, 3) or quat.shape != (n, 4) or joint_pos.shape != (n, len(names)):
        raise ValueError(f"Invalid frame array shapes in {path.name}")
    if not all(np.isfinite(a).all() for a in (pos, quat, joint_pos)):
        raise ValueError(f"Nonfinite motion values in {path.name}")
    norms = np.linalg.norm(quat, axis=1)
    if not np.allclose(norms, 1, atol=1e-3):
        raise ValueError(f"Invalid root quaternion in {path.name}")
    qpos = np.tile(model.qpos0, (n, 1))
    qpos[:, root : root + 3] = pos
    qpos[:, root + 3 : root + 7] = quat / norms[:, None]
    qpos[:, model.jnt_qposadr[hinges]] = joint_pos[:, [names.index(name) for name in expected]]
    feet = [model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    paths = np.empty((n, 3, 3))
    paths[:, 0] = pos
    data = mujoco.MjData(model)
    for i, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        paths[i, 1:] = data.xpos[feet]
    return Motion(qpos, fps, root, paths)


@dataclass
class Playback:
    position: float = 0
    playing: bool = False
    speed: float = 1
    loop: bool = True

    def advance(self, elapsed: float, fps: float, count: int) -> int:
        if self.playing:
            self.position += max(0, elapsed) * fps * self.speed
            if self.position >= count:
                if self.loop:
                    self.position %= count
                else:
                    self.position = count - 1
                    self.playing = False
        return min(int(self.position), count - 1)
