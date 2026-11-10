"""Checkpoint-driven play architecture without changing the control contract."""

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from instinct_mj.tasks.locomotion.config.g1.mild_vae_env_cfg import instinct_g1_mild_vae_env_cfg
from instinct_mj.tasks.locomotion.config.g1.mild_vae_rl_cfg import g1_mild_vae_instinct_rl_cfg
from instinct_mj.tasks.locomotion.mild.deployment import configure_inference, gait_schema
from instinct_mj.tasks.locomotion.mild.walking_mdp import gait_clock


class InferenceConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checkpoint = Path(self.directory.name, "checkpoint.pt")

    def save_metadata(self, agent, history=10, version=6, env=None):
        train_cfg = agent.to_dict()
        if env is None:
            env = instinct_g1_mild_vae_env_cfg(variant="baseline" if agent.policy.class_name == "ActorCritic" else "full")
        state = {
            "train_cfg": train_cfg,
            "mild_schema": {
                "version": version,
                "history_length": history,
                "policy": copy.deepcopy(train_cfg["policy"]),
                **gait_schema(env),
            },
        }
        torch.save(state, self.checkpoint)
        return state

    def test_nondefault_architecture_restored_without_changing_play_controls(self):
        trained = g1_mild_vae_instinct_rl_cfg("full")
        trained.policy = replace(
            trained.policy,
            history_length=4,
            latent_dim=7,
            encoder_hidden_dims=[64, 32],
            decoder_hidden_dims=[32, 64],
            actor_hidden_dims=[64],
            critic_hidden_dims=[32],
            modulation_hidden_dims=[16],
            state_scales=(2.0, 3.0, 4.0),
            init_noise_std=0.2,
            use_modulation=False,
        )
        self.save_metadata(trained, history=4)
        env = instinct_g1_mild_vae_env_cfg(play=True, variant="full")
        env.commands["base_velocity"].ranges.lin_vel_x = (0.3, 0.3)
        before = {name: copy.deepcopy(getattr(env, name)) for name in ("actions", "events", "commands", "rewards")}
        default_agent = g1_mild_vae_instinct_rl_cfg("full")
        restored_env, restored_agent = configure_inference(env, default_agent, self.checkpoint)
        self.assertEqual(restored_agent.policy, trained.policy)
        self.assertEqual(default_agent.policy.history_length, 10)
        self.assertEqual(restored_env.history_length, 4)
        self.assertEqual(restored_env.observations["policy"].history_length, 4)
        self.assertTrue(restored_env.observations["policy"].flatten_history_dim)
        self.assertFalse(restored_env.observations["policy"].enable_corruption)
        for name, value in before.items():
            self.assertEqual(getattr(restored_env, name), value, name)

    def test_baseline_architecture_restores_without_enabling_history(self):
        trained = g1_mild_vae_instinct_rl_cfg("baseline")
        trained.policy = replace(trained.policy, actor_hidden_dims=[64], critic_hidden_dims=[32])
        self.save_metadata(trained, history=1)
        env = instinct_g1_mild_vae_env_cfg(play=True, variant="baseline")
        restored_env, agent = configure_inference(env, g1_mild_vae_instinct_rl_cfg("baseline"), self.checkpoint)
        self.assertEqual(agent.policy, trained.policy)
        self.assertFalse(restored_env.use_vae)
        self.assertEqual(restored_env.observations["policy"].history_length, 0)

    def test_old_schema_and_mismatched_policy_class_are_rejected(self):
        trained = g1_mild_vae_instinct_rl_cfg("full")
        env = instinct_g1_mild_vae_env_cfg(play=True)
        for version in (4, 5):
            with self.subTest(version=version):
                state = self.save_metadata(trained, version=version)
                del state["mild_schema"]["gait"]
                del state["mild_schema"]["gait_clock_convention"]
                torch.save(state, self.checkpoint)
                with self.assertRaisesRegex(ValueError, "version-6.*gait-conditioned"):
                    configure_inference(env, trained, self.checkpoint)
        self.save_metadata(trained)
        with self.assertRaisesRegex(ValueError, "policy class"):
            configure_inference(env, g1_mild_vae_instinct_rl_cfg("baseline"), self.checkpoint)

    def test_inconsistent_history_or_policy_metadata_is_rejected(self):
        trained = g1_mild_vae_instinct_rl_cfg("full")
        env = instinct_g1_mild_vae_env_cfg(play=True)
        state = self.save_metadata(trained, history=3)
        with self.assertRaisesRegex(ValueError, "history lengths"):
            configure_inference(env, trained, self.checkpoint)
        state["mild_schema"]["history_length"] = 10
        state["train_cfg"]["policy"]["latent_dim"] = 5
        torch.save(state, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            configure_inference(env, trained, self.checkpoint)

    def test_saved_gait_restores_clock_timing_for_every_variant(self):
        for variant in ("baseline", "vae", "full"):
            with self.subTest(variant=variant):
                trained = g1_mild_vae_instinct_rl_cfg(variant)
                train_env = instinct_g1_mild_vae_env_cfg(variant=variant)
                train_env.gait_period = 0.64
                train_env.gait_stance_fraction = 0.7
                train_env.gait_clearance = 0.08
                train_env.gait_contact_threshold = 8.0
                train_env.randomize_gait_phase_by_env = False
                self.save_metadata(trained, history=1 if variant == "baseline" else 10, env=train_env)
                play_env = instinct_g1_mild_vae_env_cfg(play=True, variant=variant)
                restored, _ = configure_inference(play_env, trained, self.checkpoint)
                self.assertEqual(gait_schema(restored), gait_schema(train_env))
                self.assertEqual(restored.scene.num_envs, 1)
                self.assertFalse(restored.observations["policy"].enable_corruption)
                env = SimpleNamespace(
                    cfg=restored,
                    num_envs=2,
                    device="cpu",
                    episode_length_buf=torch.full((2,), 8),
                    step_dt=0.02,
                    command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([[1.0, 0.0, 0.0]]).expand(2, -1)),
                )
                # 8 steps are exactly one quarter of the restored 0.64 s period;
                # the default period or default per-environment offset differs.
                torch.testing.assert_close(gait_clock(env), torch.tensor([[1.0, 0.0], [1.0, 0.0]]), atol=1e-6, rtol=0)

    def test_missing_or_invalid_gait_metadata_is_rejected(self):
        trained = g1_mild_vae_instinct_rl_cfg("full")
        env = instinct_g1_mild_vae_env_cfg(play=True)
        for name, value, message in (
            ("gait_period", 0.0, "positive"),
            ("gait_period", float("nan"), "finite"),
            ("gait_clearance", -0.1, "positive"),
            ("gait_contact_threshold", float("inf"), "finite"),
            ("gait_stance_fraction", 0.5, "between"),
            ("gait_stance_fraction", 1.0, "between"),
            ("randomize_gait_phase_by_env", 1, "boolean"),
        ):
            with self.subTest(name=name, value=value):
                state = self.save_metadata(trained)
                state["mild_schema"]["gait"][name] = value
                torch.save(state, self.checkpoint)
                with self.assertRaisesRegex(ValueError, message):
                    configure_inference(env, trained, self.checkpoint)
        state = self.save_metadata(trained)
        del state["mild_schema"]["gait"]["gait_period"]
        torch.save(state, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "gait configuration.*incomplete"):
            configure_inference(env, trained, self.checkpoint)
        state = self.save_metadata(trained)
        state["mild_schema"]["gait_clock_convention"]["right_foot_phase_offset"] = 0.0
        torch.save(state, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "gait clock convention"):
            configure_inference(env, trained, self.checkpoint)


if __name__ == "__main__":
    unittest.main()
