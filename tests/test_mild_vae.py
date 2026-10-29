"""Estimator/PPO contracts, plus opt-in real GPU history/label checks."""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from instinct_rl.storage import RolloutStorage

from instinct_mj.tasks.locomotion.mild.actor_critic import MildActorCritic
from instinct_mj.tasks.locomotion.mild.ppo import MildVaePPO
from instinct_mj.tasks.locomotion.mild.vae_env import MildVaeEnv, PhysicsLatency


def observation_format(history=10):
    widths = dict(base_ang_vel=3, projected_gravity=3, velocity_commands=3, joint_pos=12, joint_vel=12, actions=12)
    return {
        "policy": {name: (width * history,) for name, width in widths.items()},
        "critic": {
            **{name: (width,) for name, width in widths.items()},
            "state_targets": (11,),
            "terrain_parameters": (22,),
            "policy_history": (45 * history,),
        },
    }


class MildVaeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.model = MildActorCritic(observation_format(), 12)
        self.history = torch.randn(8, 450)
        self.critic = torch.cat((torch.randn(8, 78), self.history), dim=-1)

    def test_current_frame_schema_and_causality(self):
        history = torch.cat(
            [torch.arange(10)[None, :, None].expand(1, 10, width).flatten(1) for width in (3, 3, 3, 12, 12, 12)], dim=-1
        ).float()
        torch.testing.assert_close(self.model.current_frame(history), torch.full((1, 45), 9.0))
        with self.assertRaises(ValueError):
            MildActorCritic(observation_format(), 12, history_length=7)

    def test_actor_does_not_read_critic_and_ppo_ratio_is_one(self):
        actions = self.model.act(self.history)
        old_log_prob = self.model.get_actions_log_prob(actions).detach()
        expected = self.model.act_inference(self.history).detach()
        modified = self.critic.clone()
        modified[:, :78] += 100
        first_value = self.model.evaluate(self.critic).detach()
        second_value = self.model.evaluate(modified).detach()
        self.assertFalse(torch.allclose(first_value, second_value))
        torch.testing.assert_close(self.model.act_inference(self.history), expected, rtol=0, atol=0)
        self.model.act(self.history)
        ratio = (self.model.get_actions_log_prob(actions) - old_log_prob).exp()
        torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=0, atol=0)

    def test_actor_gradient_reaches_estimator_and_modulation(self):
        self.model.act_inference(self.history).square().mean().backward()
        for name in ("encoder", "state_head", "latent_mean", "modulation"):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in getattr(self.model, name).parameters()), 0.0)

    def test_joint_loss_weights_and_all_five_auxiliary_terms(self):
        alg = MildVaePPO(self.model, regression_weight=0.3, vae_kl_weight=0.001, entropy_coef=0.008)
        actions = self.model.act(self.history).detach()
        minibatch = RolloutStorage.MiniBatch(
            self.history,
            self.critic,
            actions,
            self.model.evaluate(self.critic).detach(),
            torch.ones(8, 1),
            torch.zeros(8, 1),
            self.model.get_actions_log_prob(actions).detach().unsqueeze(-1),
            self.model.action_mean.detach(),
            self.model.action_std.detach(),
            None,
            None,
        )
        losses, _, stats = alg.compute_losses(minibatch)
        self.assertAlmostEqual(alg.surrogate_loss_coef, 0.7)
        self.assertAlmostEqual(alg.value_loss_coef, 0.7)
        self.assertAlmostEqual(alg.entropy_coef, 0.7 * 0.008)
        self.assertAlmostEqual(alg.regression_loss_coef, 0.3)
        reference = sum(
            stats[f"estimator/{name}"] * weight
            for name, weight in (
                ("reconstruction", 1),
                ("vae_kl", 0.001),
                ("base_velocity", 1),
                ("feet_velocity", 1),
                ("feet_height", 1),
            )
        )
        torch.testing.assert_close(losses["regression_loss"], reference)
        total = sum(value * getattr(alg, name + "_coef", 1.0) for name, value in losses.items())
        total.backward()
        self.assertGreater(self.model.decoder[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.latent_logvar.weight.grad.abs().sum().item(), 0)
        self.assertEqual(stats["ppo_ratio_deviation"].item(), 0.0)

    def test_checkpoint_roundtrip_and_onnx(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(self.model.state_dict(), path)
            restored = MildActorCritic(observation_format(), 12)
            restored.load_state_dict(torch.load(path, weights_only=True))
            expected = self.model.act_inference(self.history).detach()
            torch.testing.assert_close(restored.act_inference(self.history), expected, rtol=0, atol=0)
            restored.export_as_onnx(self.history, directory)
            import numpy as np
            import onnxruntime as ort

            session = ort.InferenceSession(str(Path(directory) / "actor.onnx"), providers=["CPUExecutionProvider"])
            actual = session.run(None, {"policy_history": self.history.numpy()})[0]
            np.testing.assert_allclose(actual, expected.numpy(), atol=1e-6, rtol=1e-5)

    def test_task_removal_and_amp_preservation(self):
        from instinct_mj.tasks.registry import list_tasks, load_env_cfg, load_instinct_rl_cfg

        self.assertNotIn("Instinct-Locomotion-Mild-G1-v0", list_tasks())
        self.assertNotIn("Instinct-Locomotion-Mild-G1-Play-v0", list_tasks())
        cfg = load_env_cfg("Instinct-Locomotion-Mild-Walk-G1-v0")
        self.assertIn("amp_policy", cfg.observations)
        self.assertIn("amp_reference", cfg.observations)
        cfg = load_env_cfg("Instinct-Locomotion-vae-G1-v0")
        self.assertEqual(set(cfg.observations), {"policy", "critic"})
        self.assertFalse(any(s.name == "motion_reference" for s in cfg.scene.sensors))
        self.assertEqual(len(cfg.rewards), 7)
        cfg.__post_init__()  # CLI reconstruction must not restore inherited rewards.
        self.assertEqual(len(cfg.rewards), 7)
        self.assertEqual(load_instinct_rl_cfg("Instinct-Locomotion-vae-G1-v0").normalizers, {})


class ControlContractTests(unittest.TestCase):
    def test_pd_samples_fresh_feedback_at_delayed_command_arrival(self):
        data = SimpleNamespace(
            joint_pos=torch.zeros(2, 3),
            joint_vel=torch.zeros(2, 3),
            joint_pos_target=torch.ones(2, 3),
            default_joint_pos=torch.zeros(2, 3),
        )
        recorded = []
        robot = SimpleNamespace(data=data, set_joint_effort_target=lambda value: recorded.append(value.clone()))
        env = SimpleNamespace(
            scene={"robot": robot},
            control_tick=0,
            num_envs=2,
            device="cpu",
            physics_dt=0.001,
            upper_ids=[2],
            kp=torch.tensor([50.0, 20.0, 0.0]),
            kd=torch.tensor([1.0, 1.0, 0.0]),
            effort_limit=torch.full((3,), 100.0),
            held_torque=torch.zeros(2, 3),
            cfg=SimpleNamespace(decimation=10, pd_decimation=10, latency_range_ms=(0.0, 3.0), motor_offset_rad=0.0),
        )
        env.latency = PhysicsLatency(env)
        env.latency.reset(torch.arange(2))
        env.latency.lags[:, 0] = torch.tensor([0, 3])
        MildVaeEnv._before_physics_step(env)
        torch.testing.assert_close(recorded[-1], torch.tensor([[50.0, 20.0, 0.0], [0.0, 0.0, 0.0]]))
        data.joint_pos.fill_(0.25)
        for _ in range(2):
            MildVaeEnv._before_physics_step(env)
            torch.testing.assert_close(recorded[-1], recorded[0])
        MildVaeEnv._before_physics_step(env)
        torch.testing.assert_close(recorded[-1], torch.tensor([[50.0, 20.0, 0.0], [37.5, 15.0, 0.0]]))
        data.joint_pos.fill_(0.5)
        for _ in range(6):
            MildVaeEnv._before_physics_step(env)
            torch.testing.assert_close(recorded[-1], recorded[3])
        MildVaeEnv._before_physics_step(env)
        torch.testing.assert_close(recorded[-1], torch.tensor([[25.0, 10.0, 0.0], [37.5, 15.0, 0.0]]))
        torch.testing.assert_close(data.joint_pos_target[:, 2], torch.zeros(2))

    def test_nominal_arms_do_not_collide_with_thighs_at_randomized_reset(self):
        import mujoco
        import numpy as np
        from mjlab.utils.lab_api.string import resolve_matching_names_values

        from instinct_mj.tasks.registry import load_env_cfg

        robot = load_env_cfg("Instinct-Locomotion-vae-G1-v0").scene.entities["robot"]
        model = robot.spec_fn().compile()
        data = mujoco.MjData(model)
        names = [model.joint(i).name for i in range(1, model.njnt)]
        ids, _, values = resolve_matching_names_values(robot.init_state.joint_pos, names)
        nominal = np.zeros(len(names))
        nominal[ids] = values
        rng = np.random.default_rng(42)
        for _ in range(128):
            data.qpos[:3] = robot.init_state.pos
            data.qpos[3:7] = [1, 0, 0, 0]
            data.qpos[7:] = nominal * rng.uniform(0.8, 1.2, len(names))
            mujoco.mj_forward(model, data)
            self.assertFalse(any(contact.dist < -1e-5 for contact in data.contact))

    def test_vae_phase_has_no_hidden_environment_index_offset(self):
        from instinct_mj.tasks.locomotion.mild.walking_mdp import foot_phase

        env = SimpleNamespace(
            num_envs=4,
            device="cpu",
            episode_length_buf=torch.full((4,), 13),
            step_dt=0.01,
            cfg=SimpleNamespace(gait_period=0.8, randomize_gait_phase_by_env=False),
        )
        phase = foot_phase(env)
        torch.testing.assert_close(phase, phase[:1].expand_as(phase))
        env.cfg.randomize_gait_phase_by_env = True
        self.assertFalse(torch.allclose(foot_phase(env), phase))


class LatencyTests(unittest.TestCase):
    def test_physics_delay_and_partial_reset_do_not_leak(self):
        data = SimpleNamespace(
            joint_pos=torch.zeros(2, 29),
            default_joint_pos=torch.zeros(2, 29),
            joint_pos_target=torch.zeros(2, 29),
            root_link_ang_vel_b=torch.zeros(2, 3),
            projected_gravity_b=torch.zeros(2, 3),
        )
        robot = SimpleNamespace(data=data, set_joint_effort_target=lambda value: data.joint_pos_target.copy_(value))
        command = torch.zeros(2, 3)
        env = SimpleNamespace(
            num_envs=2,
            device="cpu",
            physics_dt=0.001,
            scene={"robot": robot},
            cfg=SimpleNamespace(latency_range_ms=(1.0, 10.0), motor_offset_rad=0.0),
            command_manager=SimpleNamespace(get_command=lambda _: command),
        )
        delay = PhysicsLatency(env)
        delay.reset(torch.arange(2))
        delay.ensure_reset_sensors()
        delay.lags.fill_(3)
        for i in range(1, 8):
            data.joint_pos_target.fill_(i)
            data.root_link_ang_vel_b.fill_(i)
            command.fill_(i)
            delayed = delay.motor_target()
            delay.sample_sensors()
            torch.testing.assert_close(delayed, torch.full((2, 29), float(max(0, i - 3))))
            torch.testing.assert_close(delay.imu[:, :3], torch.full((2, 3), float(max(0, i - 2))))
            torch.testing.assert_close(delay.command, torch.full((2, 3), float(max(0, i - 2))))
        previous = delay.sensor_buffer[:, 1].clone()
        delay.reset(torch.tensor([0]))
        delay.ensure_reset_sensors()
        torch.testing.assert_close(previous, delay.sensor_buffer[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(delay.motor_buffer[:, 0], torch.zeros_like(delay.motor_buffer[:, 0]))


@unittest.skipUnless(
    os.environ.get("INSTINCT_MILD_GPU_TESTS") == "1" and torch.cuda.is_available(),
    "Set INSTINCT_MILD_GPU_TESTS=1 for GPU integration",
)
class VaeIntegrationTests(unittest.TestCase):
    def test_airborne_g1_leg_is_stable_with_1khz_servo(self):
        from instinct_mj.tasks.registry import load_env_cfg, load_env_cls

        task = "Instinct-Locomotion-vae-G1-Play-v0"
        cfg = load_env_cfg(task)
        cfg.scene.num_envs = 1
        cfg.auto_reset = False
        cfg.pd_decimation = 1
        cfg.latency_range_ms = (10.0, 10.0)
        cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, 3.0)
        cfg.deformable_contact.debug_vis = False
        env = load_env_cls(task)(cfg, device="cuda:0")
        self.addCleanup(env.close)
        env.reset()
        with torch.inference_mode():
            for step in range(30):
                action = torch.zeros(1, 12, device=env.device)
                action[0, 2] = 0.02 if step == 0 else 0.0
                _, _, done, timeout, _ = env.step(action)
                self.assertFalse(done.any() or timeout.any())
                self.assertLess(env.mild_contact.foot_force.abs().max().item(), 1e-6)
                self.assertLess(env.scene["robot"].data.joint_vel.abs().max().item(), 1.0)

    def test_history_partial_reset_labels_and_checkpoint_resume(self):
        from instinct_mj.tasks.locomotion.mild.vae_mdp import state_targets
        from instinct_mj.tasks.registry import (
            load_env_cfg,
            load_env_cls,
            load_instinct_rl_cfg,
            load_runner_cls,
            load_vecenv_cls,
        )

        task = "Instinct-Locomotion-vae-G1-v0"
        cfg = load_env_cfg(task)
        cfg.scene.num_envs = 2
        cfg.observations["policy"].enable_corruption = False
        env = load_env_cls(task)(cfg, device="cuda:0")
        self.addCleanup(env.close)
        wrapper = load_vecenv_cls(task)(env)
        agent = load_instinct_rl_cfg(task)
        runner = load_runner_cls(task)(wrapper, agent.to_dict(), device="cuda:0")
        self.assertEqual(env.action_manager.get_term("joint_pos").target_ids.tolist(), env.leg_ids)
        self.assertEqual(wrapper.num_obs, 450)
        self.assertEqual(wrapper.num_critic_obs, 528)
        for _ in range(5):
            obs, reward, _, info = wrapper.step(torch.zeros((2, 12), device="cuda:0"))
            self.assertTrue(torch.isfinite(reward).all())
        torch.testing.assert_close(info["observations"]["critic"][:, -450:], obs, rtol=0, atol=0)
        torch.testing.assert_close(info["observations"]["critic"][:, 45:56], state_targets(env))
        other = obs[1].clone()
        raw, _ = env.reset(env_ids=torch.tensor([0], device="cuda:0"))
        after = wrapper._pack_observations(raw)["policy"]
        torch.testing.assert_close(after[1], other, rtol=0, atol=0)
        for section, width in zip(runner.alg.actor_critic.policy_slices.values(), (3, 3, 3, 12, 12, 12)):
            history = after[0, section].reshape(10, width)
            torch.testing.assert_close(history, history[-1:].expand_as(history))
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
