"""Opt-in real GPU command/history/reset checks for both simulator environments.

INSTINCT_TELEOP_GPU_TESTS=1 .venv/bin/python -m unittest discover -s tests -p test_teleop_integration.py
"""

import os
import unittest

import torch

from instinct_mj.teleop.inputs.gamepad import ZeroInput
from instinct_mj.teleop.registry import build_env_config, get_task
from instinct_mj.teleop.runtime import TeleopSession
from instinct_mj.tasks.registry import load_env_cls, load_vecenv_cls


@unittest.skipUnless(
    os.environ.get("INSTINCT_TELEOP_GPU_TESTS") == "1" and torch.cuda.is_available(),
    "Set INSTINCT_TELEOP_GPU_TESTS=1 for MuJoCo Warp integration tests",
)
class TeleopIntegrationTests(unittest.TestCase):
    def check_profile(self, alias):
        spec = get_task(alias)
        cfg = spec.defaults()
        env_cfg = build_env_config(cfg)
        env = load_env_cls(spec.task_id)(cfg=env_cfg, device="cuda:0")
        try:
            vec = load_vecenv_cls(spec.task_id)(env)
            command = spec.adapter().bind_command(env)
            session = TeleopSession(vec, command, ZeroInput())
            actions = torch.zeros((1, vec.num_actions), device="cuda:0")
            command.set_normalized((0.5, 0.0, 0.5))
            for _ in range(8):
                session.step(actions)
                self.assertFalse(session.needs_reset)
            self.assertTrue(torch.isfinite(session.get_observations()).all())
            self.assertGreater(command.command[0, 0].item(), 0.1)
            policy_obs = env.observation_manager.compute()["policy"]
            newest_velocity = policy_obs["velocity_commands"].reshape(1, -1, 3)[:, -1]
            torch.testing.assert_close(newest_velocity, command.command)
            cached = session.get_observations().clone()
            for _ in range(5):
                session.get_observations()
                vec.get_observations()
            torch.testing.assert_close(session.get_observations(), cached)
            self.assertEqual(env.episode_length_buf.item(), 8)
            if alias == "mild_walk":
                self.assertGreater(policy_obs["gait_clock"].abs().sum().item(), 0)
                self.assertTrue(hasattr(env, "mild_contact"))
            else:
                self.assertTrue(torch.isfinite(policy_obs["depth_image"]).all())
                self.assertEqual(policy_obs["velocity_commands"].shape, (1, 24))
            session.reset()
            self.assertEqual(command.command.count_nonzero().item(), 0)
            self.assertEqual(command.target.count_nonzero().item(), 0)
            reset_obs = env.observation_manager.compute()["policy"]
            self.assertEqual(reset_obs["velocity_commands"].count_nonzero().item(), 0)
            self.assertEqual(env.episode_length_buf.item(), 0)
            session.step(actions)
            self.assertFalse(session.needs_reset)
        finally:
            env.close()

    def test_mild_command_clock_and_reset(self):
        self.check_profile("mild_walk")

    def test_parkour_command_depth_history_and_reset(self):
        self.check_profile("parkour_amp")


if __name__ == "__main__":
    unittest.main()
