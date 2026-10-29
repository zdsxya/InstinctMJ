"""Independent checks for quaternion conversion, gating, and task isolation."""

import unittest
from dataclasses import asdict
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_and_audit import audit, resample_qpos


class UmrPipelineTests(unittest.TestCase):
    def test_quaternion_sign_flip_is_not_rotation(self):
        q = np.zeros((30, 36))
        q[:, 3] = 1
        q[1::2, 3] = -1
        q[:, 0] = np.arange(30) / 30
        result = resample_qpos(q, 30)
        np.testing.assert_allclose(result[:, 3], 1)
        np.testing.assert_allclose(np.diff(result[:, 0]), .02)

    def test_ground_penetration_cannot_pass_audit(self):
        from instinct_mj.assets.unitree_g1 import G1_MJCF_PATH
        model = mujoco.MjModel.from_xml_path(G1_MJCF_PATH)
        q = np.tile(model.qpos0, (30, 1))
        q[:, 2] = .4
        result = audit(model, q, 50)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["ground_penetration_m"])

    def test_walk_task_keeps_amp_and_no_encoder(self):
        from instinct_mj.tasks.registry import load_env_cfg, load_instinct_rl_cfg
        from instinct_mj.tasks.parkour.config.g1.rl_cfg import AmpAlgoCfg
        from instinct_mj.tasks.locomotion.config.g1.mild_rl_cfg import G1MildPPORunnerCfg
        walk = load_instinct_rl_cfg("Instinct-Locomotion-Mild-Walk-G1-v0")
        mild = G1MildPPORunnerCfg()
        self.assertEqual(asdict(walk.policy), asdict(mild.policy))
        self.assertEqual(walk.algorithm.class_name, "WasabiPPO")
        self.assertEqual(mild.algorithm.class_name, "PPO")
        for key, value in vars(AmpAlgoCfg()).items():
            if key.startswith("discriminator_") and key != "discriminator_reward_coef":
                self.assertEqual(getattr(walk.algorithm, key), value)
        self.assertGreater(walk.algorithm.discriminator_reward_coef, 0)
        self.assertLess(walk.algorithm.discriminator_reward_coef, AmpAlgoCfg().discriminator_reward_coef)
        for key, value in vars(mild.algorithm).items():
            if key != "class_name" and not key.startswith("discriminator_"):
                self.assertEqual(getattr(walk.algorithm, key), value)
        env = load_env_cfg("Instinct-Locomotion-Mild-Walk-G1-v0")
        self.assertTrue(env.terminations["dataset_exhausted"].params["reset_without_notice"])


if __name__ == "__main__":
    unittest.main()
