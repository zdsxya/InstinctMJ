"""29-DoF estimator/PPO contracts and opt-in MuJoCo Warp integration."""

import copy
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from instinct_rl.algorithms.ppo import PPO
from instinct_rl.modules.actor_critic import ActorCritic
from instinct_rl.storage import RolloutStorage

from instinct_mj.tasks.locomotion.config.g1.mild_env_cfg import G1MildEnvCfg
from instinct_mj.tasks.locomotion.config.g1.mild_vae_env_cfg import instinct_g1_mild_vae_env_cfg
from instinct_mj.tasks.locomotion.config.g1.mild_vae_rl_cfg import g1_mild_vae_instinct_rl_cfg
from instinct_mj.tasks.locomotion.config.g1.mild_walk_env_cfg import instinct_g1_mild_walk_env_cfg
from instinct_mj.tasks.locomotion.mild.actor_critic import MildActorCritic
from instinct_mj.tasks.locomotion.mild.ppo import MildVaePPO
from instinct_mj.tasks.locomotion.mild.vae_env import MildVaeEnv


def observation_widths(num_actions=29):
    return dict(base_ang_vel=3, projected_gravity=3, velocity_commands=3,
                joint_pos=num_actions, joint_vel=num_actions, actions=num_actions, gait_clock=2)


def observation_format(history=10, num_actions=29, supervised=True):
    widths = observation_widths(num_actions)
    critic = {"base_lin_vel": (3,), **{name: (width,) for name, width in widths.items()}}
    if supervised:
        critic["state_targets"] = (11,)
    return {"policy": {name: (width * history,) for name, width in widths.items()}, "critic": critic}


def _read_frame(env, name):
    return env.frames[name]


def _flatten(group):
    return torch.cat(list(group.values()), dim=-1)


class MildVaeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.model = MildActorCritic(observation_format(), 29)
        self.history = torch.randn(8, 980)
        self.critic = torch.randn(8, 112)

    def test_whole_body_shapes_and_configurable_dimensions(self):
        estimate, mean, logvar = self.model.encode(self.history)
        self.assertEqual(estimate.shape, (8, 11))
        self.assertEqual(mean.shape, (8, 16))
        self.assertEqual(logvar.shape, (8, 16))
        self.assertEqual(self.model.decoder(mean).shape, (8, 98))
        self.assertEqual(self.model.actor[0].in_features, 125)
        self.assertEqual(self.model.critic[0].in_features, 101)
        self.assertEqual(self.model.act_inference(self.history).shape, (8, 29))
        model = MildActorCritic(
            observation_format(history=4, num_actions=17), 17, history_length=4,
            latent_dim=7, encoder_hidden_dims=(64, 32), decoder_hidden_dims=(32, 64),
            actor_hidden_dims=(32,), critic_hidden_dims=(32,), modulation_hidden_dims=(16,),
        )
        history = torch.randn(2, 4 * 62)
        estimate, mean, logvar = model.encode(history)
        self.assertEqual(mean.shape, (2, 7))
        self.assertEqual(model.decoder(mean).shape, (2, 62))
        self.assertEqual(model.actor[0].in_features, 80)
        self.assertEqual(model.act_inference(history).shape, (2, 17))

    def test_current_frame_and_reconstruction_do_not_target_future(self):
        history = torch.cat([
            torch.arange(10)[None, :, None].expand(1, 10, width).flatten(1)
            for width in observation_widths().values()
        ], dim=-1).float()
        current = torch.full((1, 98), 9.0)
        torch.testing.assert_close(self.model.current_frame(history), current)
        # A decoder returning exactly the current sensor frame must have zero
        # reconstruction loss, irrespective of the privileged training labels.
        hook = self.model.decoder.register_forward_hook(lambda module, inputs, result: current)
        try:
            losses = self.model.auxiliary_losses(history, torch.randn(1, 112))
            self.assertEqual(losses["reconstruction"].item(), 0.0)
        finally:
            hook.remove()
        with self.assertRaises(ValueError):
            MildActorCritic(observation_format(), 29, history_length=7)

    def test_supervision_does_not_enter_actor_or_critic(self):
        expected_actions = self.model.act_inference(self.history).detach()
        expected_value = self.model.evaluate(self.critic).detach()
        modified = self.critic.clone()
        modified[:, -11:] += 100
        torch.testing.assert_close(self.model.evaluate(modified), expected_value, rtol=0, atol=0)
        modified[:, :3] += 100
        self.assertFalse(torch.allclose(self.model.evaluate(modified), expected_value))
        torch.testing.assert_close(self.model.act_inference(self.history), expected_actions, rtol=0, atol=0)
        self.model.evaluate(self.critic).sum().backward()
        self.assertTrue(all(p.grad is None for p in self.model.estimator.parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in self.model.critic.parameters()), 0)

    def test_policy_probability_is_deterministic_across_auxiliary_sampling(self):
        actions = self.model.act(self.history).detach()
        old_log_prob = self.model.get_actions_log_prob(actions).detach()
        expected_mean = self.model.action_mean.detach().clone()
        self.model.auxiliary_losses(self.history, self.critic)
        self.model.act(self.history)
        torch.testing.assert_close(self.model.action_mean, expected_mean, rtol=0, atol=0)
        ratio = (self.model.get_actions_log_prob(actions) - old_log_prob).exp()
        torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=0, atol=0)

    def test_actor_gradient_reaches_all_state_heads_and_modulation(self):
        self.model.act_inference(self.history).square().mean().backward()
        modules = [self.model.encoder, self.model.state_head, self.model.latent_mean,
                   self.model.modulation, self.model.actor]
        for module in modules:
            with self.subTest(module=type(module).__name__):
                self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None), 0)
        for head in (self.model.estimator.base_velocity_head, self.model.estimator.feet_velocity_head,
                     self.model.estimator.feet_height_head):
            self.assertGreater(head.weight.grad.abs().sum().item(), 0)

    def test_vae_ablation_bypasses_modulation(self):
        model = MildActorCritic(observation_format(), 29, use_modulation=False)
        estimate, mean, _ = model.encode(self.history)
        torch.testing.assert_close(model.modulate(estimate, mean), mean, rtol=0, atol=0)
        self.assertIsNone(model.modulation)
        self.assertFalse(any("modulation" in name for name, _ in model.named_parameters()))
        self.assertEqual(model.act_inference(self.history).shape, (8, 29))

    def test_modulation_uses_only_feet_and_starts_near_identity(self):
        estimate, mean, _ = self.model.encode(self.history)
        scale, shift = self.model.modulation.parameters_for(estimate[:, 3:9])
        self.assertEqual(scale.shape, (8, 16))
        self.assertEqual(shift.shape, (8, 16))
        self.assertLess(scale.abs().max().item(), 0.01)
        self.assertLess(shift.abs().max().item(), 0.01)
        result = self.model.modulate(estimate, mean)
        torch.testing.assert_close(result, (1 + scale) * mean + shift)
        unrelated = estimate.clone()
        unrelated[:, :3] += 100
        unrelated[:, 9:] -= 100
        torch.testing.assert_close(self.model.modulate(unrelated, mean), result, rtol=0, atol=0)

    def test_state_scaling_keeps_estimates_in_physical_units(self):
        unscaled = copy.deepcopy(self.model)
        unscaled.state_scales.fill_(1)
        torch.testing.assert_close(self.model.encode(self.history)[0], unscaled.encode(self.history)[0])
        scaled_losses = self.model.auxiliary_losses(self.history, self.critic)
        unscaled_losses = unscaled.auxiliary_losses(self.history, self.critic)
        for name in ("base_velocity", "feet_velocity"):
            torch.testing.assert_close(scaled_losses[name], unscaled_losses[name])
        torch.testing.assert_close(scaled_losses["feet_height"], 100 * unscaled_losses["feet_height"])

    def test_joint_loss_coefficients_and_finite_gradients(self):
        alg = MildVaePPO(self.model, regression_weight=0.3, vae_kl_weight=0.001, entropy_coef=0.008)
        actions = self.model.act(self.history).detach()
        minibatch = RolloutStorage.MiniBatch(
            self.history, self.critic, actions, self.model.evaluate(self.critic).detach(),
            torch.ones(8, 1), torch.zeros(8, 1),
            self.model.get_actions_log_prob(actions).detach().unsqueeze(-1),
            self.model.action_mean.detach(), self.model.action_std.detach(), None, None,
        )
        losses, _, stats = alg.compute_losses(minibatch)
        self.assertAlmostEqual(alg.surrogate_loss_coef, 0.7)
        self.assertAlmostEqual(alg.value_loss_coef, 0.7)
        self.assertAlmostEqual(alg.entropy_coef, 0.7 * 0.008)
        self.assertAlmostEqual(alg.regression_loss_coef, 0.3)
        reference = sum(stats[f"estimator/{name}"] * weight for name, weight in (
            ("reconstruction", 1), ("vae_kl", 0.001), ("base_velocity", 1),
            ("feet_velocity", 1), ("feet_height", 1),
        ))
        torch.testing.assert_close(losses["regression_loss"], reference)
        total = sum(value * getattr(alg, name + "_coef", 1.0) for name, value in losses.items())
        self.assertTrue(torch.isfinite(total))
        total.backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(self.model.decoder[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.latent_logvar.weight.grad.abs().sum().item(), 0)
        self.assertEqual(stats["ppo_ratio_deviation"].item(), 0.0)
        custom = MildVaePPO(
            self.model, regression_weight=0.2, reconstruction_weight=0.5,
            vae_kl_weight=0.002, base_velocity_weight=2, feet_velocity_weight=3, feet_height_weight=4,
        )
        losses, _, stats = custom.compute_losses(minibatch)
        expected = sum(stats[f"estimator/{name}"] * weight for name, weight in (
            ("reconstruction", 0.5), ("vae_kl", 0.002), ("base_velocity", 2),
            ("feet_velocity", 3), ("feet_height", 4),
        ))
        torch.testing.assert_close(losses["regression_loss"], expected)
        self.assertAlmostEqual(custom.regression_loss_coef, 0.2)

    def test_checkpoint_roundtrip_and_onnx(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(self.model.state_dict(), path)
            restored = MildActorCritic(observation_format(), 29)
            restored.load_state_dict(torch.load(path, weights_only=True))
            expected = self.model.act_inference(self.history).detach()
            torch.testing.assert_close(restored.act_inference(self.history), expected, rtol=0, atol=0)
            restored.export_as_onnx(self.history, directory)
            import numpy as np
            import onnxruntime as ort

            session = ort.InferenceSession(str(Path(directory) / "actor.onnx"), providers=["CPUExecutionProvider"])
            actual = session.run(None, {"policy_history": self.history.numpy()})[0]
            np.testing.assert_allclose(actual, expected.numpy(), atol=1e-6, rtol=1e-5)


class ConfigurationTests(unittest.TestCase):
    def test_all_variants_retain_project_control_and_task(self):
        original = G1MildEnvCfg()
        walk = instinct_g1_mild_walk_env_cfg()
        gait_rewards = ("gait_contact", "swing_clearance", "both_feet_airborne")
        agents = []
        for variant in ("baseline", "vae", "full"):
            with self.subTest(variant=variant):
                cfg = instinct_g1_mild_vae_env_cfg(variant=variant)
                self.assertEqual(cfg.sim.mujoco.timestep, 0.001)
                self.assertEqual(cfg.decimation, 20)
                self.assertEqual(cfg.actions, original.actions)
                self.assertEqual(cfg.actions["joint_pos"].actuator_names, (".*",))
                self.assertEqual(cfg.scene.entities["robot"].articulation.actuators,
                                 original.scene.entities["robot"].articulation.actuators)
                self.assertEqual(cfg.scene.entities["robot"].init_state,
                                 original.scene.entities["robot"].init_state)
                for name in ("commands", "events", "terminations", "deformable_contact"):
                    self.assertEqual(getattr(cfg, name), getattr(original, name), name)
                self.assertEqual(list(cfg.observations["policy"].terms), list(observation_widths()))
                self.assertEqual(set(cfg.rewards), set(original.rewards) | set(gait_rewards))
                for name, reward in original.rewards.items():
                    self.assertEqual(cfg.rewards[name], reward, name)
                for name in gait_rewards:
                    self.assertEqual(cfg.rewards[name], walk.rewards[name], name)
                for name in ("gait_period", "gait_stance_fraction", "gait_clearance",
                             "gait_contact_threshold", "randomize_gait_phase_by_env"):
                    self.assertEqual(getattr(cfg, name), getattr(walk, name), name)
                for group in ("policy", "critic"):
                    for name, term in original.observations[group].terms.items():
                        self.assertEqual(cfg.observations[group].terms[name], term)
                    self.assertEqual(cfg.observations[group].terms["gait_clock"],
                                     walk.observations[group].terms["gait_clock"])
                self.assertEqual(set(cfg.observations), {"policy", "critic"})
                self.assertEqual("state_targets" in cfg.observations["critic"].terms, variant != "baseline")
                self.assertNotIn("terrain_parameters", cfg.observations["critic"].terms)
                self.assertFalse(any(s.name == "motion_reference" for s in cfg.scene.sensors))
                rewards = copy.deepcopy(cfg.rewards)
                cfg.__post_init__()  # tyro may reconstruct dataclass instances.
                self.assertEqual(cfg.rewards, rewards)
                self.assertEqual(cfg.actions, original.actions)
                agent = g1_mild_vae_instinct_rl_cfg(variant=variant)
                self.assertEqual(agent.normalizers, {})
                agents.append(agent)
        for attr in ("num_steps_per_env", "max_iterations", "save_interval", "seed"):
            self.assertEqual(len({getattr(a, attr) for a in agents}), 1, attr)
        for attr in ("learning_rate", "num_learning_epochs", "num_mini_batches", "gamma", "lam",
                     "clip_param", "entropy_coef", "value_loss_coef"):
            self.assertEqual(len({getattr(a.algorithm, attr) for a in agents}), 1, attr)
        self.assertEqual(agents[0].policy.class_name, "ActorCritic")
        self.assertEqual(agents[0].algorithm.class_name, "PPO")
        self.assertFalse(agents[1].policy.use_modulation)
        self.assertTrue(agents[2].policy.use_modulation)

    def test_task_entries_and_preserved_removed_aliases(self):
        from instinct_mj.tasks.registry import (
            list_tasks, load_env_cfg, load_env_cls, load_instinct_rl_cfg, load_runner_cls, load_vecenv_cls,
        )
        from instinct_mj.tasks.locomotion.mild.runner import MildBaselineRunner, MildVaeRunner

        for old_task in ("Instinct-Locomotion-Mild-G1-v0", "Instinct-Locomotion-Mild-G1-Play-v0",
                         "Instinct-Locomotion-vae-G1-v0", "Instinct-Locomotion-vae-G1-Play-v0"):
            self.assertNotIn(old_task, list_tasks())
            with self.assertRaises(KeyError):
                load_env_cfg(old_task)
        walk = load_env_cfg("Instinct-Locomotion-Mild-Walk-G1-v0")
        self.assertIn("amp_policy", walk.observations)
        self.assertIn("amp_reference", walk.observations)
        for suffix, variant in (("-Baseline", "baseline"), ("", "vae"), ("-Full", "full")):
            for play in (False, True):
                task = f"Instinct-Locomotion-Mild-VAE{suffix}-G1{'-Play' if play else ''}-v0"
                with self.subTest(task=task):
                    self.assertIn(task, list_tasks())
                    cfg = load_env_cfg(task)
                    self.assertEqual(cfg.use_vae, variant != "baseline")
                    self.assertEqual(cfg.scene.num_envs, 1 if play else 4096)
                    self.assertIs(load_env_cls(task), MildVaeEnv)
                    self.assertTrue(callable(load_vecenv_cls(task)))
                    self.assertIs(load_runner_cls(task), MildBaselineRunner if variant == "baseline" else MildVaeRunner)
                    self.assertEqual(load_instinct_rl_cfg(task).normalizers, {})
                    self.assertEqual(load_env_cfg(task, play=True).scene.num_envs, 1)


class ObservationContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_mjlab_history_alignment_and_partial_reset(self):
        from mjlab.managers import ObservationManager

        cfg = instinct_g1_mild_vae_env_cfg()
        env = SimpleNamespace(num_envs=3, device="cpu", frames={})
        groups = copy.deepcopy(cfg.observations)
        widths = {"base_lin_vel": 3, **observation_widths(), "state_targets": 11}
        for group in groups.values():
            group.enable_corruption = False
            for name, term in group.terms.items():
                term.func, term.params, term.noise, term.scale = _read_frame, {"name": name}, None, None
                env.frames[name] = torch.zeros(3, widths[name])
        manager = ObservationManager(groups, env)
        for step in range(12):
            for frame in env.frames.values():
                frame.copy_(torch.tensor([step, 100 + step, 200 + step])[:, None])
            result = manager.compute(update_history=True)
        history = _flatten(result["policy"])
        self.assertEqual(history.shape, (3, 980))
        for name, width in observation_widths().items():
            value = result["policy"][name].reshape(3, 10, width)
            torch.testing.assert_close(value[0, :, 0], torch.arange(2, 12).float())
        model = MildActorCritic(observation_format(), 29)
        torch.testing.assert_close(model.current_frame(history), torch.tensor([11, 111, 211])[:, None].expand(-1, 98).float())
        # Reading cached observations must not age or duplicate the history.
        torch.testing.assert_close(_flatten(manager.compute()["policy"]), history, rtol=0, atol=0)
        untouched = history[[0, 2]].clone()
        ids = torch.tensor([1])
        manager.reset(ids)
        for frame in env.frames.values():
            frame[1] = -7
        after = manager.compute(update_history=True, env_ids=ids)
        torch.testing.assert_close(_flatten(after["policy"])[[0, 2]], untouched, rtol=0, atol=0)
        torch.testing.assert_close(_flatten(after["policy"])[1], torch.full((980,), -7.0))
        torch.testing.assert_close(after["critic"]["state_targets"][1], torch.full((11,), -7.0))

    def test_state_targets_use_root_frame_and_signed_sole_height(self):
        from instinct_mj.tasks.locomotion.mild.vae_mdp import state_targets

        positions = torch.zeros(2, 4, 3)
        positions[0, :, 2] = torch.tensor([0.6, 0.8, 0.4, 0.6])
        positions[1, :, 2] = torch.tensor([2.4, 2.4, 2.7, 2.7])
        jacobian = torch.zeros(2, 4, 3, 1)
        jacobian[:, :, :, 0] = torch.tensor([[1., 0, 0], [3., 0, 0], [0, 2., 0], [0, 4., 0]])
        contact = SimpleNamespace(
            kinematics=lambda: (positions, jacobian), areas=torch.tensor([[1., 3.], [2., 2.]]),
            data=SimpleNamespace(qvel=torch.tensor([[1.], [2.]])), cfg=SimpleNamespace(surface_height=0.5),
        )
        robot = SimpleNamespace(data=SimpleNamespace(
            root_link_quat_w=torch.tensor([[math.sqrt(0.5), 0, 0, math.sqrt(0.5)], [1., 0, 0, 0]]),
            root_link_lin_vel_b=torch.tensor([[1., 2., 3.], [4., 5., 6.]]),
        ))
        class Scene(dict):
            env_origins = torch.tensor([[0., 0, 0], [0., 0, 2.]])
        env = SimpleNamespace(num_envs=2, device="cpu", mild_contact=contact, scene=Scene(robot=robot))
        expected = torch.tensor([[1., 2, 3, 0, -2.5, 0, 3, 0, 0, 0.25, 0],
                                 [4., 5, 6, 5, 0, 0, 0, 6, 0, -0.1, 0.2]])
        torch.testing.assert_close(state_targets(env), expected, atol=1e-6, rtol=1e-6)


class PpoRolloutTests(unittest.TestCase):
    def test_baseline_vae_full_collect_returns_and_update(self):
        """Exercise the real PPO storage/GAE/minibatches on CPU, without physics."""
        torch.set_num_threads(2)
        for variant in ("baseline", "vae", "full"):
            with self.subTest(variant=variant):
                torch.manual_seed(7)
                use_vae = variant != "baseline"
                fmt = observation_format(history=10 if use_vae else 1, supervised=use_vae)
                obs_dim, critic_dim = (980, 112) if use_vae else (98, 101)
                model = (MildActorCritic(fmt, 29, use_modulation=variant == "full") if use_vae else
                         ActorCritic(fmt, 29, actor_hidden_dims=(32,), critic_hidden_dims=(32,)))
                alg_cls = MildVaePPO if use_vae else PPO
                alg = alg_cls(model, num_learning_epochs=2, num_mini_batches=2, learning_rate=1e-3)
                alg.init_storage(4, 4, fmt, 29)
                optimizer_ids = {id(p) for group in alg.optimizer.param_groups for p in group["params"]}
                self.assertEqual(optimizer_ids, {id(p) for p in model.parameters()})
                saved_obs, saved_labels = [], []
                obs, critic = torch.randn(4, obs_dim), torch.randn(4, critic_dim)
                with torch.no_grad():
                    for step in range(4):
                        saved_obs.append(obs.clone())
                        saved_labels.append(critic.clone())
                        actions = alg.act(obs, critic)
                        self.assertEqual(actions.shape, (4, 29))
                        next_obs, next_critic = torch.randn_like(obs), torch.randn_like(critic)
                        dones = torch.zeros(4, dtype=torch.long)
                        dones[0] = int(step == 1)
                        timeouts = torch.zeros(4, dtype=torch.bool)
                        timeouts[1] = step == 3
                        infos = {"observations": {"policy": next_obs, "critic": next_critic},
                                 "step": {}, "time_outs": timeouts}
                        alg.process_env_step(torch.randn(4, 1), dones, infos, next_obs, next_critic)
                        obs, critic = next_obs, next_critic
                    alg.compute_returns(critic)
                torch.testing.assert_close(alg.storage.observations, torch.stack(saved_obs))
                torch.testing.assert_close(alg.storage.critic_observations, torch.stack(saved_labels))
                self.assertTrue(torch.isfinite(alg.storage.returns).all())
                self.assertTrue(torch.isfinite(alg.storage.advantages).all())
                minibatch = next(alg.storage.mini_batch_generator(1, 1))
                _, intermediate, _ = alg.compute_losses(minibatch)
                # Rollout batches of four and update batches of sixteen use
                # different BLAS reductions; allow float32 log-prob rounding.
                torch.testing.assert_close(intermediate["ratio"], torch.ones_like(intermediate["ratio"]), atol=1e-5, rtol=0)
                before = model.actor[0].weight.detach().clone()
                losses, stats = alg.update(0)
                self.assertEqual(alg.storage.step, 0)
                self.assertTrue(all(torch.isfinite(value).all() for value in (*losses.values(), *stats.values())))
                self.assertFalse(torch.equal(model.actor[0].weight, before))
                if use_vae:
                    self.assertIn("regression_loss", losses)
                    self.assertGreater(stats["grad/decoder"].item(), 0)
                    self.assertGreater(stats["grad/encoder"].item(), 0)
                    if variant == "full":
                        self.assertGreater(stats["grad/modulation"].item(), 0)


class RunnerContractTests(unittest.TestCase):
    @staticmethod
    def _runner(variant, directory=None):
        from instinct_mj.tasks.locomotion.mild.runner import MildBaselineRunner, MildVaeRunner

        class TensorEnvironment:
            num_envs, num_actions, num_rewards = 2, 29, 1
            physics_dt, step_dt, device = 0.001, 0.02, "cpu"

            def __init__(self):
                self.cfg = instinct_g1_mild_vae_env_cfg(variant=variant)
                self.unwrapped = self
                names = [f"joint_{i}" for i in range(29)]
                default_joint_pos = torch.linspace(-0.2, 0.2, 29).repeat(2, 1)
                self.scene = {"robot": SimpleNamespace(
                    joint_names=names, data=SimpleNamespace(default_joint_pos=default_joint_pos),
                )}
                action = SimpleNamespace(
                    target_names=names, scale=torch.full((1, 29), 0.25),
                    offset=default_joint_pos, cfg=SimpleNamespace(clip=None),
                )
                self.action_manager = SimpleNamespace(get_term=lambda _: action)

            def get_obs_format(self):
                return observation_format(history=1 if variant == "baseline" else 10,
                                          supervised=variant != "baseline")

            def reset(self):
                return torch.zeros(2, 98 if variant == "baseline" else 980), {}

        cls = MildBaselineRunner if variant == "baseline" else MildVaeRunner
        return cls(TensorEnvironment(), g1_mild_vae_instinct_rl_cfg(variant).to_dict(),
                   log_dir=directory, device="cpu")

    def test_checkpoint_restores_optimizer_schedule_counters_and_schema(self):
        torch.set_num_threads(2)
        for variant in ("baseline", "vae", "full"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                runner = self._runner(variant, directory)
                policy_obs = torch.randn(2, 98 if variant == "baseline" else 980)
                expected = runner.get_inference_policy()(policy_obs).detach().clone()
                runner.completed_updates, runner.completed_episodes = 7, 13
                runner.fallen_episodes, runner.tot_timesteps, runner.training_seconds = 3, 336, 4.5
                runner.alg.optimizer.param_groups[0]["lr"] = 1.23e-4
                path = str(Path(directory) / "checkpoint.pt")
                runner.save(path)
                metadata = torch.load(path, weights_only=True)["mild_schema"]
                self.assertEqual(metadata["version"], 6)
                self.assertEqual(metadata["gait"]["gait_period"], runner.env.cfg.gait_period)
                self.assertEqual(metadata["gait_clock_convention"]["right_foot_phase_offset"], 0.5)
                self.assertIn("gait_clock", metadata["obs_format"]["policy"])
                with torch.no_grad():
                    runner.alg.actor_critic.actor[0].weight.add_(2)
                runner.alg.learning_rate = 0.5
                runner.alg.optimizer.param_groups[0]["lr"] = 0.5
                runner.load(path)
                self.assertEqual(runner.current_learning_iteration, 7)
                self.assertEqual(runner.completed_updates, 7)
                self.assertEqual(runner.completed_episodes, 13)
                self.assertEqual(runner.fallen_episodes, 3)
                self.assertEqual(runner.tot_timesteps, 336)
                self.assertEqual(runner.training_seconds, 4.5)
                self.assertEqual(runner.alg.learning_rate, 1.23e-4)
                self.assertEqual(runner.alg.optimizer.param_groups[0]["lr"], 1.23e-4)
                torch.testing.assert_close(runner.get_inference_policy()(policy_obs), expected, rtol=0, atol=0)
                runner.schema["action_names"].reverse()
                with self.assertRaisesRegex(ValueError, "schema"):
                    runner.load(path)
                runner.schema["action_names"].reverse()
                runner.schema["gait"]["gait_period"] *= 2
                with self.assertRaisesRegex(ValueError, "schema"):
                    runner.load(path)
                runner.schema["gait"]["gait_period"] /= 2
                runner.schema["resolved_action_offset"][0] += 0.1
                with self.assertRaisesRegex(ValueError, "schema"):
                    runner.load(path)

    @staticmethod
    def _log_inputs():
        return {
            "collection_time": 0.2, "learn_time": 0.1,
            "losses": {"surrogate_loss": torch.tensor(0.2), "value_loss": torch.tensor(0.3),
                       "entropy": torch.tensor(-0.4), "total_loss": torch.tensor(0.5)},
            "stats": {"VAE/Foot_Height_MSE": torch.tensor(0.02),
                      "VAE/Foot_Height_Scaled_MSE": torch.tensor(2.0)},
            "ep_infos": [], "step_infos": [{
                "Evaluation/Velocity_Tracking_Error": torch.tensor(0.3),
                "Evaluation/Fallen_Episodes": torch.tensor(1, dtype=torch.int64),
                "Evaluation/Completed_Episodes": torch.tensor(2, dtype=torch.int64),
            }],
            "rewbuffer": [[2.0]], "lenbuffer": [5], "rframebuffer": [[0.4]],
        }

    def test_logging_keeps_tag_names_units_and_completed_update_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self._runner("full", directory)
            recorded = {}
            runner.writer = SimpleNamespace(add_scalar=lambda key, value, step: recorded.update({key: (value, step)}))
            runner.alg.actor_critic.act(torch.zeros(2, 980))
            runner.completed_episodes, runner.fallen_episodes = 2, 1
            runner.log(self._log_inputs())
            self.assertEqual(runner.completed_updates, 1)
            self.assertEqual(runner.tot_timesteps, 48)
            self.assertEqual(recorded["Evaluation/Fall_Rate"], (0.5, 1))
            self.assertEqual(recorded["Evaluation/Episode_Reward"], (2.0, 1))
            self.assertEqual(recorded["Evaluation/Episode_Length"], (5, 1))
            self.assertEqual(recorded["Evaluation/Fallen_Episodes"], (1.0, 1))
            self.assertEqual(recorded["Evaluation/Completed_Episodes"], (2.0, 1))
            self.assertAlmostEqual(recorded["VAE/Foot_Height_MSE"][0], 0.02)
            self.assertEqual(recorded["VAE/Foot_Height_Scaled_MSE"], (2.0, 1))
            self.assertFalse(any(key.startswith("Train/VAE/") for key in recorded))

    def test_periodic_checkpoint_name_matches_completed_updates_after_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self._runner("full", directory)
            runner.writer = SimpleNamespace(add_scalar=lambda *args: None)
            runner.checkpoint_interval = 2
            runner.alg.actor_critic.act(torch.zeros(2, 980))
            runner.log(self._log_inputs())
            self.assertFalse(list(Path(directory).glob("model_*.pt")))
            runner.current_learning_iteration = 1
            runner.log(self._log_inputs())
            path = Path(directory) / "model_2.pt"
            state = torch.load(path, weights_only=True)
            self.assertEqual(state["iter"], 2)
            self.assertEqual(state["policy_transitions"], 96)
            runner.load(str(path))
            for index in (2, 3):
                runner.current_learning_iteration = index
                runner.log(self._log_inputs())
            self.assertEqual(sorted(p.name for p in Path(directory).glob("model_*.pt")),
                             ["model_2.pt", "model_4.pt"])
            state = torch.load(Path(directory) / "model_4.pt", weights_only=True)
            self.assertEqual(state["iter"], 4)
            self.assertEqual(state["policy_transitions"], 192)


@unittest.skipUnless(
    os.environ.get("INSTINCT_MILD_GPU_TESTS") == "1" and torch.cuda.is_available(),
    "Set INSTINCT_MILD_GPU_TESTS=1 for GPU integration",
)
class VaeIntegrationTests(unittest.TestCase):
    def test_whole_body_actions_history_labels_and_checkpoint(self):
        from instinct_mj.tasks.locomotion.mild.runner import MildVaeRunner
        from instinct_mj.tasks.locomotion.mild.vae_mdp import state_targets
        from instinct_mj.tasks.locomotion.mild.vae_wrapper import MildVaeVecEnvWrapper
        from instinct_mj.tasks.locomotion.mild.walking_mdp import gait_clock

        cfg = instinct_g1_mild_vae_env_cfg(play=True)
        cfg.scene.num_envs = 2
        cfg.deformable_contact.debug_vis = False
        env = MildVaeEnv(cfg, device="cuda:0")
        self.addCleanup(env.close)
        wrapper = MildVaeVecEnvWrapper(env)
        agent = g1_mild_vae_instinct_rl_cfg(variant="full")
        runner = MildVaeRunner(wrapper, agent.to_dict(), device="cuda:0")
        action_term = env.action_manager.get_term("joint_pos")
        self.assertEqual(sorted(action_term.target_ids.tolist()), list(range(29)))
        self.assertEqual(wrapper.num_actions, 29)
        self.assertEqual(wrapper.num_obs, 980)
        self.assertEqual(wrapper.num_critic_obs, 112)
        for _ in range(5):
            obs, reward, _, info = wrapper.step(torch.zeros((2, 29), device="cuda:0"))
            self.assertTrue(torch.isfinite(reward).all())
        for name in ("Walk/phase_agreement", "Walk/single_support_fraction", "Walk/flight_fraction"):
            self.assertIn(name, info["step"])
            self.assertTrue(torch.isfinite(info["step"][name]).all())
        clock_section = runner.alg.actor_critic.policy_slices["gait_clock"]
        torch.testing.assert_close(obs[:, clock_section].reshape(2, 10, 2)[:, -1], gait_clock(env))
        torch.testing.assert_close(info["observations"]["critic"][:, -11:], state_targets(env))
        other = obs[1].clone()
        raw, _ = env.reset(env_ids=torch.tensor([0], device="cuda:0"))
        after = wrapper._pack_observations(raw)["policy"]
        torch.testing.assert_close(after[1], other, rtol=0, atol=0)
        torch.testing.assert_close(after[:, clock_section].reshape(2, 10, 2)[:, -1], gait_clock(env))
        for section, width in zip(runner.alg.actor_critic.policy_slices.values(), observation_widths().values()):
            history = after[0, section].reshape(10, width)
            torch.testing.assert_close(history, history[-1:].expand_as(history))
        # Upper-body commands must reach the native servo, rather than being
        # overwritten by the old twelve-action implementation's pose hold.
        upper = next(i for i, name in enumerate(action_term.target_names) if "shoulder_pitch" in name)
        actions = torch.zeros((2, 29), device="cuda:0")
        actions[:, upper] = 0.1
        wrapper.step(actions)
        joint_id = action_term.target_ids[upper]
        robot = env.scene["robot"]
        self.assertTrue((robot.data.joint_pos_target[:, joint_id] != robot.data.default_joint_pos[:, joint_id]).all())
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "checkpoint.pt")
            runner.completed_updates = 7
            runner.save(path)
            expected = runner.get_inference_policy()(after).detach()
            with torch.no_grad():
                runner.alg.actor_critic.actor[0].weight.add_(1)
            runner.load(path)
            self.assertEqual(runner.current_learning_iteration, 7)
            torch.testing.assert_close(runner.get_inference_policy()(after), expected, rtol=0, atol=0)
            runner.schema["joint_names"] = list(reversed(runner.schema["joint_names"]))
            with self.assertRaises(ValueError):
                runner.load(path)


if __name__ == "__main__":
    unittest.main()
