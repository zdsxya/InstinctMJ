"""Regression checks for data coordinates, joint mapping and playback timing.

Run: .venv/bin/python -m unittest discover -s scripts/umr -p test_viewer.py -v
"""

import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
from viewer_data import DEFAULT_XML, ROOT, MotionLibrary, Playback, load_motion


class MotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))

    def write_motion(self, path, reverse=False, invalid=False):
        hinges = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        names = [self.model.joint(int(j)).name for j in hinges]
        values = np.tile(np.linspace(-0.1, 0.1, len(names)), (3, 1))
        if reverse:
            names, values = names[::-1], values[:, ::-1]
        if invalid:
            names[0] = "wrong_joint"
        np.savez(
            path,
            framerate=50,
            joint_names=names,
            joint_pos=values,
            base_pos_w=[[1, 2, 0.8], [1.1, 2, 0.85], [1.2, 2.1, 0.8]],
            base_quat_w=np.tile([np.sqrt(0.5), 0, 0, np.sqrt(0.5)], (3, 1)),
        )

    def test_joint_order_is_mapped_by_name(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "a.npz", Path(temp) / "b.npz"
            self.write_motion(first)
            self.write_motion(second, reverse=True)
            a, b = load_motion(first, self.model), load_motion(second, self.model)
            np.testing.assert_allclose(a.qpos, b.qpos)
            np.testing.assert_allclose(a.paths, b.paths)

    def test_in_place_preserves_height_rotation_and_relative_feet(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a.npz"
            self.write_motion(path)
            motion = load_motion(path, self.model)
            data = mujoco.MjData(self.model)
            feet = [self.model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
            for i in range(3):
                pose = motion.pose(i, True)
                np.testing.assert_allclose(pose[:2], 0)
                np.testing.assert_allclose(pose[2:], motion.qpos[i, 2:])
                data.qpos[:] = pose
                mujoco.mj_forward(self.model, data)
                np.testing.assert_allclose(data.xpos[feet], motion.trajectories(True)[i, 1:], atol=1e-12)
            np.testing.assert_allclose(motion.qpos[0, :2], [1, 2])

    def test_wrong_joint_names_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.npz"
            self.write_motion(path, invalid=True)
            with self.assertRaisesRegex(ValueError, "Joint names"):
                load_motion(path, self.model)

    def test_local_dataset_matches_existing_renderer(self):
        directory = ROOT / "outputs/umr/motions"
        if not directory.exists():
            self.skipTest("Local UMR dataset is not distributed with the repository")
        library = MotionLibrary(directory)
        for name, path in library.files.items():
            with self.subTest(name=name), np.load(path) as source:
                motion = library.load(name)
                reference = np.concatenate((source["base_pos_w"], source["base_quat_w"], source["joint_pos"]), axis=1)
                actual_data, reference_data = mujoco.MjData(library.model), mujoco.MjData(library.model)
                for frame in (0, len(reference) // 2, len(reference) - 1):
                    actual_data.qpos[:] = motion.pose(frame)
                    reference_data.qpos[:] = reference[frame]
                    mujoco.mj_forward(library.model, actual_data)
                    mujoco.mj_forward(library.model, reference_data)
                    np.testing.assert_allclose(actual_data.geom_xpos, reference_data.geom_xpos, atol=1e-7)
                    np.testing.assert_allclose(actual_data.geom_xmat, reference_data.geom_xmat, atol=1e-7)


class PlaybackTests(unittest.TestCase):
    def test_timing_depends_on_elapsed_time_not_render_count(self):
        regular = Playback(playing=True, speed=2)
        delayed = Playback(playing=True, speed=2)
        for _ in range(10):
            regular.advance(0.02, 50, 149)
        self.assertEqual(delayed.advance(0.2, 50, 149), int(regular.position))

    def test_loop_and_non_loop_end(self):
        looping = Playback(position=148, playing=True)
        self.assertEqual(looping.advance(0.04, 50, 149), 1)
        stopping = Playback(position=148, playing=True, loop=False)
        self.assertEqual(stopping.advance(0.04, 50, 149), 148)
        self.assertFalse(stopping.playing)
        self.assertEqual(stopping.advance(5, 50, 149), 148)

    def test_single_frame(self):
        playback = Playback(playing=True)
        self.assertEqual(playback.advance(10, 50, 1), 0)


if __name__ == "__main__":
    unittest.main()
