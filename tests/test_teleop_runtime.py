"""CPU-only teleoperation state-machine tests, including viewer catch-up steps."""

from collections import deque
from contextlib import redirect_stdout
from dataclasses import replace
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from instinct_mj.teleop.commands.velocity import ManualVelocityCommandCfg
from instinct_mj.teleop.config import ControlConfig, TeleopConfig
from instinct_mj.teleop.inputs.gamepad import InputState
from instinct_mj.teleop.runtime import TeleopSession
from instinct_mj.teleop.viewers import TeleopHeadlessViewer, make_viewer


class FakeInput:
    def __init__(self):
        self.frames = deque()
        self.last = InputState()

    def send(self, **kwargs):
        self.frames.append(InputState(**kwargs))

    def poll(self):
        if self.frames:
            self.last = self.frames.popleft()
            return self.last
        return replace(self.last, pause=False, reset=False, quit=False)


class FakeVecEnv:
    def __init__(self):
        self.num_envs = 1
        self.device = "cpu"
        self.step_dt = 0.02
        self.cfg = SimpleNamespace(viewer=SimpleNamespace())
        self.unwrapped = self
        self.scene = {"robot": SimpleNamespace(data=SimpleNamespace(
            root_link_lin_vel_b=torch.zeros(1, 3), root_link_ang_vel_b=torch.zeros(1, 3),
        ))}
        self.reads = self.steps = self.resets = self.history_updates = 0
        self.observations = torch.zeros(1, 3)
        self.command = None
        self.terminate_at = None
        self.closed = False

    def get_observations(self):
        self.reads += 1
        return self.observations, {}

    def step(self, actions):
        self.steps += 1
        self.history_updates += 1
        self.command.compute(self.step_dt)
        self.observations = self.command.command.clone()
        done = self.terminate_at is not None and self.steps >= self.terminate_at
        return self.observations, torch.zeros(1), torch.tensor([done]), {}

    def reset(self):
        self.resets += 1
        self.history_updates += 1
        self.command.reset(torch.tensor([0]))
        self.observations = torch.zeros(1, 3)
        return self.observations, {}

    def close(self):
        self.closed = True


class FakePolicy:
    def __init__(self):
        self.seen = []
        self.resets = 0

    def __call__(self, observations):
        self.seen.append(observations.clone())
        return torch.zeros(1, 2)

    def reset(self):
        self.resets += 1


class TeleopRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(redirect_stdout(io.StringIO()))
        self.vec_env = FakeVecEnv()
        self.command = ManualVelocityCommandCfg(control=ControlConfig()).build(self.vec_env)
        self.vec_env.command = self.command
        self.source = FakeInput()
        self.policy = FakePolicy()
        self.session = TeleopSession(self.vec_env, self.command, self.source)
        self.viewer = make_viewer("none", self.session, self.policy, TeleopConfig(start_paused=True))

    def start(self):
        self.viewer.poll_input()
        self.viewer.resume()
        self.assertFalse(self.viewer._is_paused)

    def catch_up(self, seconds=0.2):
        # A constant clock makes the frame deadline deterministic. This also
        # forces all catch-up iterations through the termination/step-limit gates.
        with patch("mjlab.viewer.base.time.perf_counter", return_value=100.0):
            self.viewer._step_physics(seconds)

    def test_factory_initial_state_and_single_realtime_speed(self):
        self.assertIsInstance(self.viewer, TeleopHeadlessViewer)
        self.assertTrue(self.viewer._is_paused)
        self.assertEqual(self.viewer._time_multiplier, 1.0)
        self.viewer.increase_speed()
        self.viewer.decrease_speed()
        self.assertEqual(self.viewer._time_multiplier, 1.0)
        self.assertIs(self.viewer.env, self.session)

    def test_polling_rendering_and_observation_reads_do_not_advance_history(self):
        original = self.session.get_observations()
        for _ in range(5):
            self.viewer.poll_input()
            self.viewer.sync_env_to_viewer()
            self.assertIs(self.session.get_observations(), original)
        self.assertEqual(self.vec_env.reads, 1)
        self.assertEqual(self.vec_env.history_updates, 0)
        self.start()
        self.source.send(velocity=(1.0, 0.0, 0.0))
        self.viewer.poll_input()
        self.assertTrue(self.viewer._execute_step())
        self.assertEqual(self.vec_env.reads, 1)
        self.assertEqual(self.vec_env.history_updates, 1)
        torch.testing.assert_close(self.policy.seen[0], original)
        self.assertIs(self.session.get_observations(), self.vec_env.observations)
        self.assertGreater(self.session.get_observations()[0, 0].item(), 0)

    def test_pause_resume_uses_edges_and_requires_center_after_pause(self):
        self.start()
        self.source.send(velocity=(1.0, 0.0, 0.0))
        self.viewer.poll_input()
        self.assertGreater(self.command.target[0, 0].item(), 0)
        self.source.send(velocity=(1.0, 0.0, 0.0), pause=True)
        self.viewer.poll_input()
        self.assertTrue(self.viewer._is_paused)
        self.assertTrue(self.session.needs_neutral)
        self.assertEqual(self.command.target.count_nonzero().item(), 0)
        self.viewer.poll_input()  # Holding a button does not emit another edge.
        self.assertTrue(self.viewer._is_paused)
        self.source.send(velocity=(1.0, 0.0, 0.0), pause=True)
        self.viewer.poll_input()
        self.assertFalse(self.viewer._is_paused)
        self.assertEqual(self.command.target.count_nonzero().item(), 0)
        self.source.send()
        self.viewer.poll_input()
        self.assertFalse(self.session.needs_neutral)
        self.source.send(velocity=(0.5, 0.0, 0.0))
        self.viewer.poll_input()
        self.assertGreater(self.command.target[0, 0].item(), 0)

    def test_disconnect_stops_and_reconnect_does_not_resume(self):
        self.start()
        self.source.send(velocity=(1.0, 0.0, 1.0))
        self.viewer.poll_input()
        self.viewer._execute_step()
        self.source.send(connected=False, error="device removed")
        self.viewer.poll_input()
        self.assertTrue(self.viewer._is_paused)
        self.assertFalse(self.session.connected)
        self.assertEqual(self.session.input_error, "device removed")
        self.assertEqual(self.command.command.count_nonzero().item(), 0)
        self.assertEqual(self.command.target.count_nonzero().item(), 0)
        self.viewer.resume()
        self.assertTrue(self.viewer._is_paused)
        self.source.send(velocity=(1.0, 0.0, 0.0))
        self.viewer.poll_input()
        self.assertTrue(self.viewer._is_paused)
        self.assertTrue(self.session.needs_neutral)
        self.assertIsNone(self.session.input_error)
        self.source.send()
        self.viewer.poll_input()
        self.assertTrue(self.viewer._is_paused)
        self.source.send(pause=True)
        self.viewer.poll_input()
        self.assertFalse(self.viewer._is_paused)

    def test_reset_clears_policy_and_command_then_pauses(self):
        self.start()
        self.source.send(velocity=(1.0, 0.0, 0.0))
        self.viewer.poll_input()
        self.viewer._execute_step()
        self.session.needs_reset = True
        self.source.send(velocity=(1.0, 0.0, 0.0), reset=True)
        self.viewer.poll_input()
        self.assertEqual(self.vec_env.resets, 1)
        self.assertEqual(self.policy.resets, 1)
        self.assertFalse(self.session.needs_reset)
        self.assertTrue(self.session.needs_neutral)
        self.assertTrue(self.viewer._is_paused)
        self.assertEqual(self.command.command.count_nonzero().item(), 0)
        self.assertEqual(self.command.target.count_nonzero().item(), 0)
        self.assertIs(self.session.observations, self.vec_env.observations)
        self.assertEqual(self.session.total_steps, 1)  # Limit counts the complete session.

    def test_simultaneous_reset_and_pause_preserves_reset_pause(self):
        self.start()
        self.source.send(reset=True, pause=True)
        self.viewer.poll_input()
        self.assertEqual(self.vec_env.resets, 1)
        self.assertTrue(self.viewer._is_paused)

    def test_termination_stops_catchup_and_requires_reset(self):
        self.start()
        self.vec_env.terminate_at = 1
        self.source.send(velocity=(1.0, 0.0, 0.0))
        self.viewer.poll_input()
        self.catch_up()
        self.assertEqual(self.vec_env.steps, 1)
        self.assertEqual(self.session.total_steps, 1)
        self.assertEqual(len(self.policy.seen), 1)
        self.assertTrue(self.session.needs_reset)
        self.assertTrue(self.viewer._is_paused)
        self.assertEqual(self.viewer._sim_budget, 0.0)
        self.assertEqual(self.command.target.count_nonzero().item(), 0)
        self.viewer.resume()
        self.assertTrue(self.viewer._is_paused)
        self.catch_up()
        self.assertEqual(self.vec_env.steps, 1)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            self.session.step(torch.zeros(1, 2))

    def test_max_steps_is_strict_within_catchup_and_across_reset(self):
        self.viewer = make_viewer("none", self.session, self.policy, TeleopConfig(max_steps=3))
        self.start()
        self.catch_up()
        self.assertEqual(self.vec_env.steps, 3)
        self.assertEqual(self.session.total_steps, 3)
        self.viewer.reset_environment()
        self.viewer.poll_input()
        self.viewer.resume()
        self.catch_up()
        self.assertEqual(self.vec_env.steps, 3)
        self.assertEqual(len(self.policy.seen), 3)

    def test_quit_clears_command_and_prevents_further_steps(self):
        self.start()
        self.source.send(velocity=(1.0, 0.0, 0.0))
        self.viewer.poll_input()
        self.source.send(quit=True)
        self.viewer.poll_input()
        self.assertTrue(self.session.quit_requested)
        self.assertTrue(self.viewer._is_paused)
        self.assertEqual(self.command.target.count_nonzero().item(), 0)
        self.catch_up()
        self.assertEqual(self.vec_env.steps, 0)

    def test_viewer_queued_reset_also_clears_policy_state(self):
        self.start()
        self.viewer.request_reset()
        with patch("mjlab.viewer.base.time.perf_counter", return_value=100.0):
            self.viewer.tick()
        self.assertTrue(self.viewer._is_paused)
        self.assertEqual(self.vec_env.resets, 1)
        self.assertEqual(self.policy.resets, 1)
        self.assertEqual(self.vec_env.steps, 0)


if __name__ == "__main__":
    unittest.main()
