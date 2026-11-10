"""Configuration-only CLI checks, independent of a GPU or gamepad."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from instinct_mj.teleop.cli import main


class TeleopCliTests(unittest.TestCase):
    def test_task_defaults_cli_overrides_and_config_snapshot(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as output:
            path = Path(directory) / "config.json"
            main([
                "Instinct-Parkour-Target-Amp-G1-Play-v0", "--print-config", "True",
                "--control.lin-vel-x", "(0.0, 0.6)", "--gamepad.yaw-axis", "2",
                "--camera.follow", "False", "--dump-config", str(path),
            ])
            snapshot = json.loads(output.getvalue())
            self.assertEqual(snapshot, json.loads(path.read_text()))
            self.assertEqual(snapshot["settings"]["task"], "parkour_amp")
            self.assertEqual(snapshot["settings"]["control"]["lin_vel_x"], [0.0, 0.6])
            self.assertEqual(snapshot["settings"]["gamepad"]["yaw_axis"], 2)
            self.assertFalse(snapshot["settings"]["camera"]["follow"])
            self.assertIn("camera", snapshot["environment"]["sensors"])
            self.assertTrue(Path(snapshot["sources"]["task_defaults"]).is_file())

    def test_unadapted_task_rejected(self):
        with self.assertRaisesRegex(SystemExit, "Unsupported teleop task"):
            main(["unknown_task"])


if __name__ == "__main__":
    unittest.main()
