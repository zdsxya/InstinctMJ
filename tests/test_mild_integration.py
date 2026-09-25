"""Opt-in GPU contract checks: INSTINCT_MILD_GPU_TESTS=1 python -m unittest discover -s tests."""

import os
import unittest

import mujoco
import numpy as np
import torch

from instinct_mj.tasks.registry import load_env_cfg, load_env_cls


@unittest.skipUnless(os.environ.get("INSTINCT_MILD_GPU_TESTS") == "1" and torch.cuda.is_available(),
                     "Set INSTINCT_MILD_GPU_TESTS=1 to run MuJoCo Warp integration tests")
class MildIntegrationTests(unittest.TestCase):
    def _env(self, graph=True, task="Instinct-Locomotion-Mild-G1-Play-v0"):
        cfg = load_env_cfg(task)
        cfg.seed = 42
        cfg.scene.num_envs = 2
        cfg.deformable_contact.use_cuda_graph = graph
        env = load_env_cls(task)(cfg, device="cuda:0")
        self.addCleanup(env.close)
        env.reset()
        return env

    def test_dynamics_contact_reset_and_graph_equivalence(self):
        env = self._env()
        c = env.mild_contact
        model = env.sim.mj_model
        reference = mujoco.MjData(model)
        reference.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
        reference.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
        mujoco.mj_forward(model, reference)
        points, jacobian = c.kinematics()
        for cell in (0, c.points_per_foot - 1, c.points_per_foot, c.count - 1):
            jp = np.zeros((3, model.nv))
            mujoco.mj_jac(model, reference, jp, None, points[0, cell].cpu().numpy(),
                          c.body_ids[cell // c.points_per_foot])
            np.testing.assert_allclose(jacobian[0, cell].cpu().numpy(), jp, atol=2e-6)
        c.mass_matrix[:, c.mass_rows, c.mass_cols] = c.data.M
        c.mass_matrix[:, c.mass_cols, c.mass_rows] = c.data.M
        full_mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, reference, full_mass)
        np.testing.assert_allclose(c.mass_matrix[0].cpu().numpy(), full_mass, atol=2e-5, rtol=1e-4)

        actions = torch.zeros((2, 29), device="cuda:0")
        previous_step_info = None
        for _ in range(20):
            obs, reward, _, _, extras = env.step(actions)
            self.assertTrue(torch.isfinite(reward).all())
            if previous_step_info is not None:
                self.assertIsNot(extras["step"], previous_step_info)
            previous_step_info = extras["step"]
        self.assertGreater(c.depth.max().item(), .005)
        self.assertGreater(c.foot_force[:, :, 2].sum(-1).mean().item(), 100)
        self.assertLess(c.normal_residual.max().item(), .1)
        graph_qpos = env.sim.data.qpos.clone()

        # Same generalized contact impulse must equal the applied COM wrench.
        points, jacobian = c.kinematics()
        # Recompute at the current state, without integrating, for matching J/r.
        c.clear_applied_forces()
        env.sim.forward()
        c.step()
        points, jacobian = c.kinematics()
        generalized = torch.einsum("bcin,bci->bn", jacobian, c.forces)
        # CPU Jacobians at body COM for the same current state.
        reference.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
        reference.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
        mujoco.mj_forward(model, reference)
        result = np.zeros(model.nv)
        for foot, body in enumerate(c.body_ids):
            mujoco.mj_applyFT(model, reference, c.foot_force[0, foot].cpu().numpy(),
                             c.foot_torque[0, foot].cpu().numpy(), reference.xipos[body], body, result)
        np.testing.assert_allclose(generalized[0].cpu().numpy(), result, atol=3e-4, rtol=1e-4)

        # Native solver must not support a foot intersecting the surface.
        for contact in reference.contact:
            bodies = {int(model.geom_bodyid[g]) for g in contact.geom}
            plane = any(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE for g in contact.geom)
            self.assertFalse(plane and any(body in bodies for body in c.body_ids))

        path = c.path_length[1].clone()
        force = c.forces[1].clone()
        env.reset(env_ids=torch.tensor([0], device="cuda:0"))
        torch.testing.assert_close(c.path_length[1], path, rtol=0, atol=0)
        torch.testing.assert_close(c.forces[1], force, rtol=0, atol=0)
        self.assertEqual(c.path_length[0].sum().item(), 0)
        self.assertEqual(c.forces[0].abs().sum().item(), 0)

        eager = self._env(graph=False)
        for _ in range(20):
            eager.step(actions)
        torch.testing.assert_close(eager.sim.data.qpos.clone(), graph_qpos, rtol=1e-4, atol=1e-5)

    def test_walk_preserves_dynamics_and_resets_observable_clock(self):
        base = self._env()
        walk = self._env(task="Instinct-Locomotion-Mild-Walk-G1-Play-v0")
        initial_clock = walk.observation_manager.compute()["policy"]["gait_clock"].clone()
        actions = torch.zeros((2, 29), device="cuda:0")
        for _ in range(20):
            base.step(actions)
            obs, reward, _, _, extras = walk.step(actions)
            self.assertTrue(torch.isfinite(reward).all())
        # Separate Warp GPU simulations can differ by floating-point reduction
        # ordering; require matching trajectories and physical forces, not bits.
        torch.testing.assert_close(walk.sim.data.qpos.clone(), base.sim.data.qpos.clone(), rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(walk.mild_contact.forces, base.mild_contact.forces, rtol=1e-4, atol=1e-3)
        self.assertEqual(obs["policy"]["gait_clock"].shape, (2, 2))
        self.assertIn("Walk/phase_agreement", extras["step"])
        clock_other = obs["policy"]["gait_clock"][1].clone()
        walk.reset(env_ids=torch.tensor([0], device="cuda:0"))
        after = walk.observation_manager.compute()["policy"]["gait_clock"]
        torch.testing.assert_close(after[0], initial_clock[0])
        torch.testing.assert_close(after[1], clock_other)


if __name__ == "__main__":
    unittest.main()
