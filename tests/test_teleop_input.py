"""Joystick packet handling and externally driven command lifecycle tests."""

import errno
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from instinct_mj.teleop.commands.velocity import ManualVelocityCommandCfg
from instinct_mj.teleop.inputs.gamepad import GamepadConfig, LinuxGamepad, ZeroInput, apply_deadzone


class GamepadTests(unittest.TestCase):
    def setUp(self):
        self.packets = []
        self.open_mock = self.enterContext(patch("instinct_mj.teleop.inputs.gamepad.os.open", return_value=23))
        self.close_mock = self.enterContext(patch("instinct_mj.teleop.inputs.gamepad.os.close"))
        self.enterContext(patch("instinct_mj.teleop.inputs.gamepad.os.read", side_effect=self.read))
        self.gamepad = LinuxGamepad(GamepadConfig())
        self.addCleanup(self.gamepad.close)

    def read(self, *_):
        if not self.packets:
            raise BlockingIOError()
        item = self.packets.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def event(self, value, kind, number):
        return struct.pack("<IhBB", 0, value, kind, number)

    def test_axis_mapping_and_held_stick_without_events(self):
        self.packets.append(self.event(-32768, 0x82, 1) + self.event(32767, 0x02, 3))
        first = self.gamepad.poll()
        self.assertEqual(first.velocity, (1.0, 0.0, -1.0))
        self.assertTrue(first.connected)
        self.assertEqual(self.gamepad.poll().velocity, first.velocity)

    def test_center_deadzone_and_rescaling(self):
        self.assertEqual(apply_deadzone(0.05, 0.1), 0.0)
        self.assertEqual(apply_deadzone(-0.1, 0.1), 0.0)
        self.assertAlmostEqual(apply_deadzone(0.55, 0.1), 0.5)
        self.assertAlmostEqual(apply_deadzone(-0.55, 0.1), -0.5)
        self.assertEqual(apply_deadzone(1.0, 0.1), 1.0)

    def test_initial_button_state_does_not_trigger_action(self):
        self.packets.append(self.event(1, 0x81, 7))
        self.assertFalse(self.gamepad.poll().pause)
        self.packets.append(self.event(1, 0x01, 7))
        self.assertFalse(self.gamepad.poll().pause)
        self.packets.append(self.event(0, 0x01, 7) + self.event(1, 0x01, 7))
        self.assertTrue(self.gamepad.poll().pause)
        self.assertFalse(self.gamepad.poll().pause)

    def test_buttons_are_independent_rising_edges(self):
        self.packets.append(self.event(1, 0x01, 0) + self.event(1, 0x01, 6))
        state = self.gamepad.poll()
        self.assertTrue(state.reset)
        self.assertTrue(state.quit)
        self.assertFalse(state.pause)
        self.assertFalse(self.gamepad.poll().reset)

    def test_partial_packets_preserve_axis_value(self):
        packet = self.event(-32767, 0x02, 1)
        self.packets.append(packet[:3])
        self.assertEqual(self.gamepad.poll().velocity, (0.0, 0.0, 0.0))
        self.packets.append(packet[3:])
        self.assertEqual(self.gamepad.poll().velocity, (1.0, 0.0, 0.0))

    def test_disconnect_clears_velocity_actions_and_reconnect_state(self):
        self.packets.append(self.event(-32767, 0x02, 1))
        self.gamepad.poll()
        self.packets.extend([self.event(1, 0x01, 0), OSError(errno.ENODEV, "removed")])
        state = self.gamepad.poll()
        self.assertFalse(state.connected)
        self.assertEqual(state.velocity, (0.0, 0.0, 0.0))
        self.assertFalse(state.reset)
        self.gamepad._next_open = 0.0
        self.assertEqual(self.gamepad.poll().velocity, (0.0, 0.0, 0.0))
        self.assertEqual(self.open_mock.call_count, 2)

    def test_missing_device_retries_without_blocking(self):
        self.open_mock.side_effect = FileNotFoundError(errno.ENOENT, "missing")
        state = self.gamepad.poll()
        self.assertFalse(state.connected)
        self.assertIn("missing", state.error)
        self.assertFalse(self.gamepad.poll().connected)
        self.assertEqual(self.open_mock.call_count, 1)

    def test_eof_and_close(self):
        self.packets.append(b"")
        self.assertFalse(self.gamepad.poll().connected)
        self.gamepad.close()
        self.gamepad._next_open = 0.0
        self.assertFalse(self.gamepad.poll().connected)
        self.assertEqual(self.open_mock.call_count, 1)

    def test_invalid_config_and_zero_input(self):
        for kwargs in ({"deadzone": 1.0}, {"deadzone": float("nan")}, {"forward_sign": 0},
                       {"pause_button": -1}, {"reset_button": 7}, {"reconnect_interval": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GamepadConfig(**kwargs)
        state = ZeroInput().poll()
        self.assertTrue(state.connected)
        self.assertEqual(state.velocity, (0.0, 0.0, 0.0))


class ManualCommandTests(unittest.TestCase):
    def setUp(self):
        from instinct_mj.teleop.config import ControlConfig

        self.env = SimpleNamespace(num_envs=2, device="cpu", step_dt=0.02)
        self.control = ControlConfig(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0),
            linear_acceleration=1.0, yaw_acceleration=2.0,
        )
        self.command = ManualVelocityCommandCfg(control=self.control).build(self.env)

    def test_normalized_input_respects_asymmetric_limits_and_unsupported_sideways(self):
        self.command.set_normalized((1.0, 1.0, -1.0))
        torch.testing.assert_close(self.command.target, torch.tensor([[1.0, 0.0, -1.0]]).repeat(2, 1))
        self.command.set_normalized((-1.0, -1.0, 1.0))
        torch.testing.assert_close(self.command.target, torch.tensor([[-0.5, 0.0, 1.0]]).repeat(2, 1))

    def test_acceleration_is_sim_time_based_and_has_no_overshoot(self):
        self.command.set_target((1.0, 0.0, -1.0))
        self.assertEqual(self.command.command.count_nonzero().item(), 0)
        self.command.compute(0.02)
        torch.testing.assert_close(self.command.command, torch.tensor([[0.02, 0.0, -0.04]]).repeat(2, 1))
        for _ in range(100):
            self.command.compute(0.02)
        torch.testing.assert_close(self.command.command, self.command.target)
        self.command.set_target((0.0, 0.0, 0.0))
        self.command.compute(0.02)
        torch.testing.assert_close(self.command.command, torch.tensor([[0.98, 0.0, -0.96]]).repeat(2, 1))

    def test_stop_reset_and_subset_reset_clear_targets(self):
        self.command.set_target((0.5, 0.0, 0.8))
        self.command.compute(1.0)
        self.command.reset(torch.tensor([0]))
        torch.testing.assert_close(self.command.command[0], torch.zeros(3))
        torch.testing.assert_close(self.command.target[0], torch.zeros(3))
        torch.testing.assert_close(self.command.command[1], torch.tensor([0.5, 0.0, 0.8]))
        self.command.stop()
        self.command.compute(1.0)
        self.assertEqual(self.command.command.count_nonzero().item(), 0)

    def test_per_environment_dt_and_no_automatic_resampling(self):
        self.command.set_target((0.5, 0.0, 0.5))
        self.command.time_left.fill_(-1000)
        self.command.compute(torch.tensor([0.0, 0.1]))
        torch.testing.assert_close(self.command.command, torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.2]]))
        self.command.compute(0.1, env_ids=torch.tensor([0]))
        torch.testing.assert_close(self.command.command, torch.tensor([[0.1, 0.0, 0.2], [0.1, 0.0, 0.2]]))

    def test_clamps_physical_input_and_rejects_invalid_input(self):
        self.command.set_target((100.0, -10.0, -100.0))
        torch.testing.assert_close(self.command.target[0], torch.tensor([1.0, 0.0, -1.0]))
        for values in ((float("nan"), 0, 0), (float("inf"), 0, 0), (1, 2)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.command.set_target(values)
        for dt in (-0.1, float("nan"), torch.tensor([0.1])):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                self.command.compute(dt)

    def test_mild_gait_clock_reads_the_same_applied_command(self):
        from instinct_mj.tasks.locomotion.mild.walking_mdp import gait_clock

        self.env.cfg = SimpleNamespace(gait_period=0.8, randomize_gait_phase_by_env=False)
        self.env.episode_length_buf = torch.ones(2)
        self.env.command_manager = SimpleNamespace(get_command=lambda _: self.command.command)
        self.command.set_target((0.5, 0.0, 0.0))
        self.assertEqual(gait_clock(self.env).count_nonzero().item(), 0)
        self.command.compute(0.2)
        self.assertGreater(gait_clock(self.env).count_nonzero().item(), 0)
        self.command.on_viewer_pause(True)
        self.assertEqual(gait_clock(self.env).count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
