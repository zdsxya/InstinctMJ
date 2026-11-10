"""Task preparation contracts, without constructing a GPU simulation."""

import json
import math
import unittest

import instinct_mj.tasks  # noqa: F401
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_runner_cls
from instinct_mj.teleop.config import describe_config
from instinct_mj.teleop.registry import build_env_config, get_task, list_tasks


class TeleopAdapterTests(unittest.TestCase):
    def test_explicit_profile_resolution_preserves_task_classes(self):
        self.assertEqual(list_tasks(), ["mild_walk", "parkour_amp", "parkour_supermimic"])
        for name in list_tasks():
            task = get_task(name)
            self.assertIs(get_task(task.task_id), task)
            self.assertIs(get_task(task.task_id.removesuffix("-v0") + "-Play-v0"), task)
            self.assertEqual(task.defaults().task, name)
        self.assertEqual(load_env_cls(get_task("mild_walk").task_id).__name__, "MildWalkRlEnv")
        self.assertIsNone(load_runner_cls(get_task("parkour_amp").task_id))
        self.assertEqual(load_runner_cls(get_task("parkour_supermimic").task_id).__name__, "ParkourSuperMimicRunner")
        with self.assertRaisesRegex(ValueError, "Unsupported teleop task"):
            get_task("some_other_parkour_task")

    def test_mild_keeps_contact_actions_and_clock_without_training_data(self):
        task = get_task("mild_walk")
        base = load_env_cfg(task.task_id, play=False)
        cfg = task.defaults()
        cfg.scene.spawn_offset = (0.1, -0.2, 0.03)
        cfg.scene.spawn_yaw = 0.5
        env = build_env_config(cfg)
        self.assertEqual(env.sim.mujoco.timestep, base.sim.mujoco.timestep)
        self.assertEqual(env.decimation, base.decimation)
        self.assertEqual(env.actions, base.actions)
        self.assertIn("gait_clock", env.observations["policy"].terms)
        self.assertFalse(env.deformable_contact.randomize_material)
        self.assertEqual(env.events["reset_robot_joints"].params["position_range"], (1.0, 1.0))
        self.assertEqual(env.events["reset_base"].params["pose_range"]["z"], (0.03, 0.03))
        self.assertEqual(env.events["reset_base"].params["pose_range"]["yaw"], (0.5, 0.5))
        self.assertIn("fallen", env.terminations)
        self.assertIn("excessive_penetration", env.terminations)
        self._assert_interactive_only(env)
        # No in-place edit to the original training or play factories.
        self.assertTrue(base.deformable_contact.randomize_material)
        self.assertIn("amp_reference", base.observations)
        self.assertEqual(load_env_cfg(task.task_id, play=True).commands["base_velocity"].ranges.lin_vel_x, (1.6, 1.6))

    def test_parkour_keeps_actor_and_critic_history_camera_and_spawn_contract(self):
        for name in ("parkour_amp", "parkour_supermimic"):
            task = get_task(name)
            base = load_env_cfg(task.task_id, play=False)
            cfg = task.defaults()
            cfg.scene.terrain = "pyramid_stairs"
            cfg.scene.terrain_difficulty = 0.4
            env = build_env_config(cfg)
            self._assert_interactive_only(env)
            for group_name in ("policy", "critic"):
                expected = base.observations[group_name].terms
                actual = env.observations[group_name].terms
                self.assertEqual(list(actual), list(expected))
                for term_name in expected:
                    self.assertEqual(actual[term_name].func, expected[term_name].func)
                    self.assertEqual(actual[term_name].history_length, expected[term_name].history_length)
                    self.assertEqual(actual[term_name].scale, expected[term_name].scale)
                    self.assertEqual(actual[term_name].params, expected[term_name].params)
            camera = next(s for s in env.scene.sensors if s.name == "camera")
            original_camera = next(s for s in base.scene.sensors if s.name == "camera")
            self.assertEqual(camera.data_histories, original_camera.data_histories)
            self.assertEqual(camera.noise_pipeline, original_camera.noise_pipeline)
            self.assertEqual(env.scene.entities["robot"].init_state, base.scene.entities["robot"].init_state)
            self.assertEqual(env.events["reset_robot_joints"].params["position_range"], (0.0, 0.0))
            self.assertEqual(set(env.terminations), {"terrain_out_bound", "base_contact", "bad_orientation", "root_height"})
            generator = env.scene.terrain.terrain_generator
            self.assertFalse(generator.curriculum)
            self.assertEqual(list(generator.sub_terrains), ["pyramid_stairs"])
            self.assertEqual(generator.difficulty_range, (0.4, 0.4))
            self.assertEqual((generator.num_rows, generator.num_cols), (1, 1))
            self.assertIsNone(generator.sub_terrains["pyramid_stairs"].flat_patch_sampling)
            self.assertEqual(env.sim.mujoco.timestep, base.sim.mujoco.timestep)
            self.assertEqual(env.decimation, base.decimation)
            self.assertEqual(env.actions, base.actions)

    def test_config_validation_and_independent_defaults(self):
        task = get_task("mild_walk")
        changed = task.defaults()
        changed.control.lin_vel_x = (-0.2, 0.3)
        self.assertEqual(task.defaults().control.lin_vel_x, (-0.5, 1.0))
        for field, bad in (("lin_vel_x", (0.1, 1.0)), ("lin_vel_y", (0, math.nan))):
            cfg = task.defaults()
            setattr(cfg.control, field, bad)
            with self.assertRaises(ValueError):
                cfg.validate()
        cfg = task.defaults()
        cfg.control.yaw_acceleration = 0
        with self.assertRaises(ValueError):
            cfg.validate()
        cfg = task.defaults()
        cfg.max_steps = 0
        with self.assertRaises(ValueError):
            cfg.validate()
        for name in list_tasks():
            cfg = get_task(name).defaults()
            cfg.scene.terrain = "not_a_terrain"
            with self.assertRaisesRegex(ValueError, "terrain"):
                build_env_config(cfg)

    def test_resolved_settings_include_sources_and_actual_policy_timing(self):
        cfg = get_task("mild_walk").defaults()
        report = describe_config(cfg, build_env_config(cfg))
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["environment"]["policy_dt"], 0.02)
        self.assertTrue(report["sources"]["task_defaults"].endswith("teleop/tasks/mild_walk/config.py"))
        self.assertTrue(report["sources"]["base_task"].endswith("mild_walk_env_cfg.py"))

    def _assert_interactive_only(self, env):
        self.assertEqual(env.scene.num_envs, 1)
        self.assertFalse(env.auto_reset)
        self.assertEqual(set(env.observations), {"policy", "critic"})
        self.assertTrue(all(not group.enable_corruption for group in env.observations.values()))
        self.assertNotIn("motion_reference", [sensor.name for sensor in env.scene.sensors])
        self.assertEqual(env.rewards, {})
        self.assertEqual(env.curriculum, {})
        self.assertEqual(set(env.events), {"reset_base", "reset_robot_joints"})
        self.assertNotIn("time_out", env.terminations)
        self.assertNotIn("dataset_exhausted", env.terminations)
        self.assertEqual(type(env.commands["base_velocity"]).__name__, "ManualVelocityCommandCfg")


if __name__ == "__main__":
    unittest.main()
