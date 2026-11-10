"""Inference contracts without a simulator, training dataset, or GPU."""

import contextlib
import copy
import io
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from instinct_rl import modules

from instinct_mj.tasks.registry import load_instinct_rl_cfg
from instinct_mj.teleop.policies.loader import ZeroPolicy, load_policy


class TeleopPolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checkpoint = Path(self.directory.name) / "model_10.pt"
        self.config = {
            "policy": {
                "class_name": "ActorCritic",
                "actor_hidden_dims": [8],
                "critic_hidden_dims": [8],
                "activation": "elu",
            },
            "normalizers": {"policy": {"class_name": "EmpiricalNormalization"}},
        }
        self.env = SimpleNamespace(
            get_obs_format=lambda: {"policy": {"proprio": (7,)}, "critic": {"proprio": (9,)}},
            num_actions=2,
            num_rewards=1,
        )
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.task_patch = patch(
            "instinct_mj.teleop.policies.loader.load_instinct_rl_cfg",
            side_effect=lambda _task: SimpleNamespace(to_dict=lambda: copy.deepcopy(self.config)),
        )
        self.task_patch.start()
        self.addCleanup(self.task_patch.stop)

    def _save(self, config=None, env=None):
        config = copy.deepcopy(config or self.config)
        env = env or self.env
        policy_config = config["policy"]
        actor = modules.build_actor_critic(
            policy_config.pop("class_name"), policy_config, env.get_obs_format(), env.num_actions, env.num_rewards
        ).eval()
        # Optimizer, discriminator, training state and datasets are intentionally absent.
        checkpoint = {"model_state_dict": actor.state_dict(), "iter": 10}
        normalizer = None
        if "policy" in config["normalizers"]:
            dim = sum(math.prod(shape) for shape in env.get_obs_format()["policy"].values())
            normalizer = modules.EmpiricalNormalization(dim)
            normalizer.update(torch.randn((20, dim)) * 3 + 2)
            normalizer.eval()
            checkpoint["policy_normalizer_state_dict"] = normalizer.state_dict()
        torch.save(checkpoint, self.checkpoint)
        return actor, normalizer

    def test_normalized_actor_matches_training_inference_and_statistics_stay_frozen(self):
        actor, normalizer = self._save()
        policy = load_policy("test-task", self.checkpoint, self.env, "cpu")
        observations = torch.randn((3, 7))
        count_before = policy.normalizer.count.clone()
        torch.testing.assert_close(policy(observations), actor.act_inference(normalizer(observations)))
        torch.testing.assert_close(policy.normalizer.count, count_before)
        self.assertFalse(policy.actor_critic.training)
        self.assertFalse(policy.normalizer.training)
        self.assertFalse(policy(observations).requires_grad)

    def test_missing_required_normalization_fails_instead_of_changing_policy(self):
        self._save()
        state = torch.load(self.checkpoint, weights_only=True)
        state.pop("policy_normalizer_state_dict")
        torch.save(state, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "missing.*policy_normalizer_state_dict"):
            load_policy("test-task", self.checkpoint, self.env, "cpu")

    def test_unexpected_normalization_is_not_silently_discarded(self):
        self._save()
        self.config["normalizers"] = {}
        with self.assertRaisesRegex(ValueError, "contains a policy observation normalizer"):
            load_policy("test-task", self.checkpoint, self.env, "cpu")

    def test_wrong_task_shape_reports_actionable_error(self):
        self._save()
        wrong_env = SimpleNamespace(**vars(self.env))
        wrong_env.num_actions = 3
        with self.assertRaisesRegex(ValueError, "does not match task.*test-task"):
            load_policy("test-task", self.checkpoint, wrong_env, "cpu")

    def test_bad_checkpoint_format_and_onnx_are_rejected(self):
        torch.save({"wm": {}}, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "expected 'model_state_dict'"):
            load_policy("test-task", self.checkpoint, self.env, "cpu")
        with self.assertRaisesRegex(ValueError, "ONNX is not supported"):
            load_policy("test-task", self.checkpoint.with_suffix(".onnx"), self.env, "cpu")

    def test_recurrent_state_resets_to_first_step(self):
        self.config["policy"].update(
            class_name="ActorCriticRecurrent", rnn_type="gru", rnn_hidden_size=5, rnn_num_layers=1
        )
        self._save()
        policy = load_policy("test-task", self.checkpoint, self.env, "cpu")
        observations = torch.ones((1, 7))
        first = policy(observations).clone()
        policy(observations)
        policy.reset()
        torch.testing.assert_close(policy(observations), first)

    def test_observation_and_nonfinite_action_errors_precede_simulation(self):
        self._save()
        policy = load_policy("test-task", self.checkpoint, self.env, "cpu")
        with self.assertRaisesRegex(ValueError, "expects observations"):
            policy(torch.zeros((1, 8)))
        with self.assertRaisesRegex(FloatingPointError, "nonfinite actions"):
            policy(torch.full((1, 7), float("nan")))

    def test_zero_policy_preserves_batch_and_action_dimensions(self):
        policy = ZeroPolicy(29, "cpu")
        actions = policy(torch.ones((2, 98)))
        self.assertEqual(actions.shape, (2, 29))
        self.assertEqual(actions.count_nonzero().item(), 0)

    def test_registered_mild_and_both_parkour_architectures_load_without_runners(self):
        self.task_patch.stop()
        mild_policy = {
            "base_ang_vel": (3,), "projected_gravity": (3,), "velocity_commands": (3,),
            "joint_pos": (29,), "joint_vel": (29,), "actions": (29,), "gait_clock": (2,),
        }
        parkour_policy = {
            "base_ang_vel": (24,), "projected_gravity": (24,), "velocity_commands": (24,),
            "joint_pos": (232,), "joint_vel": (232,), "actions": (232,), "depth_image": (8, 18, 32),
        }
        for task_id, segments, critic_velocity_shape in (
            ("Instinct-Locomotion-Mild-Walk-G1-v0", mild_policy, (3,)),
            ("Instinct-Parkour-Target-Amp-G1-v0", parkour_policy, (24,)),
            ("Instinct-Parkour-Target-Amp-SuperMimic-G1-v0", parkour_policy, (24,)),
        ):
            with self.subTest(task_id=task_id):
                env = SimpleNamespace(
                    get_obs_format=lambda: {
                        "policy": segments, "critic": {"base_lin_vel": critic_velocity_shape, **segments},
                    },
                    num_actions=29,
                    num_rewards=1,
                )
                self._save(load_instinct_rl_cfg(task_id).to_dict(), env)
                policy = load_policy(task_id, self.checkpoint, env, "cpu")
                dim = sum(math.prod(shape) for shape in segments.values())
                self.assertEqual(policy(torch.zeros((1, dim))).shape, (1, 29))


if __name__ == "__main__":
    unittest.main()
