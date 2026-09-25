"""Mechanics invariants; run with python -m unittest discover -s tests."""

import unittest
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from instinct_mj.tasks.locomotion.mild.config import MildContactCfg
from instinct_mj.tasks.locomotion.mild.geometry import FOOT_NAMES, mild_g1_spec, sole_quadrature
from instinct_mj.tasks.locomotion.mild.solver import granular_law, solve_contact


class MildMechanicsTests(unittest.TestCase):
    def setUp(self):
        cfg = MildContactCfg()
        self.material = {name: torch.tensor([[(bounds[0] + bounds[1]) / 2]], dtype=torch.float64)
                         for name, bounds in vars(cfg.material).items()}
        self.reference_area = cfg.reference_cell_size**2

    def test_force_and_mass_are_positive_and_grid_independent(self):
        d = torch.tensor([[0.0, 0.01, 0.03]], dtype=torch.float64)
        q, k, m, drag = granular_law(d, -torch.ones_like(d), torch.ones_like(d), self.material, self.reference_area)
        self.assertTrue(torch.all(q[:, 1:] > q[:, :-1]))
        self.assertTrue(torch.all(m[:, 1:] > m[:, :-1]))
        self.assertEqual(q[0, 0], 0)
        self.assertEqual(m[0, 0], 0)
        # Splitting the same physical area into 4 quadrature points must NOT
        # multiply the material stiffness, mass or impact response by four.
        one = granular_law(d[:, 1:2], -torch.ones((1, 1)), torch.ones((1, 1)), self.material, self.reference_area)
        four = granular_law(d[:, 1:2].expand(1, 4), -torch.ones((1, 4)), torch.full((1, 4), .25), self.material, self.reference_area)
        for first, split in zip(one, four):
            torch.testing.assert_close(first.sum(), split.sum())
        h = 1e-6
        upper = granular_law(d + h, -torch.ones_like(d), torch.ones_like(d), self.material, self.reference_area)[0]
        torch.testing.assert_close((upper - q) / h, k, rtol=5e-4, atol=1e-3)

    def test_no_recruitment_drag_when_withdrawing(self):
        drag = granular_law(torch.tensor([[.03]]), torch.tensor([[1.0]]), torch.ones((1, 1)),
                            self.material, self.reference_area)[-1]
        self.assertEqual(drag.item(), 0)

    def _solve(self, speed=0.0, active=True):
        # Unit-mass point in x/z, one contact with 100 N static support.
        matrix = torch.eye(2, dtype=torch.float64)[None]
        jac = torch.tensor([[[[1., 0.], [0., 0.], [0., 1.]]]], dtype=torch.float64)
        return solve_contact(matrix, jac, torch.tensor([[speed, 0.]], dtype=torch.float64),
                             torch.tensor([[0., -100.]], dtype=torch.float64),
                             torch.zeros((1, 1, 3), dtype=torch.float64),
                             torch.tensor([[100.]], dtype=torch.float64),
                             torch.tensor([[2.]], dtype=torch.float64),
                             torch.zeros((1, 1, 2), dtype=torch.float64),
                             torch.tensor([[.5]], dtype=torch.float64),
                             torch.tensor([[active]]), torch.zeros((1, 1, 2), dtype=torch.float64),
                             .01, 60, 3)

    def test_static_support_and_implicit_added_mass(self):
        force, _, _, residual = self._solve()
        torch.testing.assert_close(force, torch.tensor([[[0., 0., 100.]]], dtype=torch.float64))
        self.assertLess(residual.item(), 1e-8)

    def test_friction_cone_and_dissipation(self):
        force, impulse, residual, _ = self._solve(speed=2.0)
        self.assertLess(force[0, 0, 0], 0)
        self.assertLessEqual(impulse.norm().item(), force[0, 0, 2].item() * .5 * .01 + 1e-9)
        self.assertLess(residual.item(), 1e-8)
        # Below the static limit, stop without reversing velocity.
        force, _, _, _ = self._solve(speed=.1)
        self.assertAlmostEqual(.1 + .01 * force[0, 0, 0].item(), 0, places=7)

    def test_no_support_in_air(self):
        force, impulse, _, _ = self._solve(speed=2, active=False)
        self.assertEqual(force.abs().sum().item(), 0)
        self.assertEqual(impulse.abs().sum().item(), 0)

    def test_geometry_area_convergence_and_collision_masks(self):
        model = mild_g1_spec().compile()
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FOOT_NAMES[0])
        coarse, areas = sole_quadrature(model, body, .0195, 16)
        fine, fine_areas = sole_quadrature(model, body, .00975, 16)
        self.assertTrue(20 < len(coarse) < 100)
        self.assertGreater(len(fine), len(coarse))
        self.assertLess(abs(areas.sum() / fine_areas.sum() - 1), .02)
        self.assertTrue(.008 < areas.sum() < .02)
        self.assertTrue(np.all(coarse[:, 2] < -.025))
        foot_geoms = np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype != 0))
        for gid in foot_geoms:
            self.assertFalse((model.geom_contype[gid] & 1) or (model.geom_conaffinity[gid] & 1))
            self.assertTrue((model.geom_contype[gid] & 3) or (model.geom_conaffinity[gid] & 1))

    def test_flat_and_mild_agent_schema_match(self):
        from instinct_mj.tasks.registry import load_env_cfg, load_instinct_rl_cfg

        flat = load_env_cfg("Instinct-Locomotion-Flat-G1-v0")
        mild = load_env_cfg("Instinct-Locomotion-Mild-G1-v0")
        for group in ("policy", "critic"):
            self.assertEqual(list(flat.observations[group].terms), list(mild.observations[group].terms))
        self.assertEqual(flat.decimation * flat.sim.mujoco.timestep, mild.decimation * mild.sim.mujoco.timestep)
        for name, term in flat.rewards.items():
            self.assertEqual(term.weight, mild.rewards[name].weight)
        a = load_instinct_rl_cfg("Instinct-Locomotion-Flat-G1-v0").to_dict()
        b = load_instinct_rl_cfg("Instinct-Locomotion-Mild-G1-v0").to_dict()
        a.pop("experiment_name")
        b.pop("experiment_name")
        self.assertEqual(a, b)

    def test_reject_signed_or_invalid_material(self):
        cfg = MildContactCfg()
        cfg.material.mass_scale = (-1, 2)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_resume_run_can_be_set_from_cli(self):
        import mjlab
        import tyro

        from instinct_mj.tasks.locomotion.config.g1.mild_rl_cfg import G1MildPPORunnerCfg

        cfg = tyro.cli(G1MildPPORunnerCfg, args=["--load-run", "/tmp/mild-resume-test"], config=mjlab.TYRO_FLAGS)
        self.assertEqual(cfg.load_run, "/tmp/mild-resume-test")
        self.assertTrue(cfg.resume)


class MildWalkingTests(unittest.TestCase):
    def test_gait_acceptance_rejects_hopping_and_static_single_support(self):
        from instinct_mj.scripts.instinct_rl.evaluate_mild_walk import summarize

        data = np.zeros((1000, 1, 15), dtype=np.float32)
        phase = np.arange(1000) * .02 / .8
        support = np.stack((phase % 1 < .6, (phase + .5) % 1 < .6), axis=-1)
        data[:, 0, :2] = support * 350
        data[:, 0, 2:4] = np.where(support, -.03, .05)
        data[:, 0, 4] = data[:, 0, 7] = .6
        data[:, 0, 12] = (np.arange(1000) + 1) * .02
        data[:, 0, 13:15] = support
        self.assertTrue(summarize(data, dt=.02)["passed"])
        data[:, 0, 2:4] = -.01
        self.assertFalse(summarize(data, dt=.02)["passed"])
        data[:, 0, :2] = (phase % 1 < .5)[:, None] * 175
        hopping = summarize(data, dt=.02)
        self.assertFalse(hopping["passed"])
        self.assertGreater(hopping["flight_fraction"], .4)
        data[:, 0, :2] = [350, 0]
        self.assertFalse(summarize(data, dt=.02)["passed"])

    def _env(self):
        from instinct_mj.tasks.locomotion.config.g1.mild_walk_env_cfg import G1MildWalkEnvCfg

        return SimpleNamespace(
            cfg=G1MildWalkEnvCfg(), num_envs=1, device="cpu", step_dt=0.02,
            episode_length_buf=torch.tensor([10]),
            command_manager=SimpleNamespace(get_command=lambda _: torch.tensor([[0.6, 0., 0.]])),
            scene=SimpleNamespace(env_origins=torch.zeros(1, 3)),
            mild_contact=SimpleNamespace(
                foot_force=torch.tensor([[[0., 0., 350.], [0., 0., 0.]]]),
                positions=torch.tensor([[[0., 0., -0.03], [0., 0., 0.06]]]),
                cfg=SimpleNamespace(surface_height=0.0),
            ),
        )

    def test_walking_prefers_alternation_over_hopping_and_dragging(self):
        from instinct_mj.tasks.locomotion.mild import walking_mdp as mdp

        env = self._env()
        self.assertEqual(mdp.desired_support(env).tolist(), [[True, False]])
        correct = mdp.phase_contact(env).item()
        clearance = mdp.swing_clearance(env).item()
        env.mild_contact.foot_force[:, :, 2] = torch.tensor([[175., 175.]])
        self.assertGreater(correct, mdp.phase_contact(env).item() + 0.4)
        env.mild_contact.foot_force.zero_()
        self.assertEqual(mdp.flight(env).item(), 1)
        self.assertGreater(correct, mdp.phase_contact(env).item() + 0.4)
        env.mild_contact.positions[:, 1, 2] = -0.01
        self.assertGreater(clearance, mdp.swing_clearance(env).item() + 0.8)
        env.episode_length_buf.fill_(30)
        self.assertEqual(mdp.desired_support(env).tolist(), [[False, True]])

    def test_clock_periodicity_no_planned_flight_and_standing(self):
        from instinct_mj.tasks.locomotion.mild import walking_mdp as mdp

        env = self._env()
        first = mdp.gait_clock(env)
        env.episode_length_buf += 40
        torch.testing.assert_close(first, mdp.gait_clock(env), atol=1e-6, rtol=0)
        for step in range(40):
            env.episode_length_buf.fill_(step)
            self.assertTrue(mdp.desired_support(env).any())
        env.command_manager.get_command = lambda _: torch.zeros(1, 3)
        self.assertEqual(mdp.gait_clock(env).abs().sum(), 0)
        self.assertTrue(mdp.desired_support(env).all())
        self.assertEqual(mdp.swing_clearance(env), 0)

    def test_walk_keeps_contact_physics_and_separates_checkpoints(self):
        from dataclasses import asdict
        from instinct_mj.tasks.registry import load_env_cfg, load_instinct_rl_cfg

        mild = load_env_cfg("Instinct-Locomotion-Mild-G1-v0")
        walk = load_env_cfg("Instinct-Locomotion-Mild-Walk-G1-v0")
        self.assertEqual(asdict(mild.deformable_contact), asdict(walk.deformable_contact))
        self.assertEqual(mild.sim.mujoco.timestep, walk.sim.mujoco.timestep)
        self.assertEqual(mild.decimation, walk.decimation)
        self.assertEqual(mild.scene.entities["robot"].spec_fn, walk.scene.entities["robot"].spec_fn)
        self.assertNotIn("gait_contact", mild.rewards)
        for group in ("policy", "critic"):
            self.assertNotIn("gait_clock", mild.observations[group].terms)
            self.assertIn("gait_clock", walk.observations[group].terms)
        self.assertNotEqual(load_instinct_rl_cfg("Instinct-Locomotion-Mild-G1-v0").experiment_name,
                            load_instinct_rl_cfg("Instinct-Locomotion-Mild-Walk-G1-v0").experiment_name)


if __name__ == "__main__":
    unittest.main()
