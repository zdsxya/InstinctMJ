"""Clock-conditioned rewards distinguish alternating support from one-foot stance."""

import unittest
from types import SimpleNamespace

import torch

from instinct_mj.tasks.locomotion.mild import walking_mdp


class GaitRewardTests(unittest.TestCase):
    def make_env(self, steps=(8, 28), command=(0.5, 0.0, 0.0)):
        count = len(steps)
        commands = torch.tensor(command).repeat(count, 1)
        return SimpleNamespace(
            num_envs=count,
            device="cpu",
            step_dt=0.02,
            episode_length_buf=torch.tensor(steps),
            cfg=SimpleNamespace(
                gait_period=0.8,
                gait_stance_fraction=0.6,
                gait_clearance=0.065,
                gait_contact_threshold=5.0,
                randomize_gait_phase_by_env=False,
            ),
            command_manager=SimpleNamespace(get_command=lambda _: commands),
            scene=SimpleNamespace(env_origins=torch.zeros(count, 3)),
            mild_contact=SimpleNamespace(
                foot_force=torch.zeros(count, 2, 3),
                positions=torch.zeros(count, 2, 3),
                cfg=SimpleNamespace(surface_height=0.0),
            ),
        )

    def test_support_switches_feet_after_half_cycle_and_rewards_follow(self):
        env = self.make_env()
        torch.testing.assert_close(
            walking_mdp.desired_support(env),
            torch.tensor([[True, False], [False, True]]),
        )
        clock = walking_mdp.gait_clock(env)
        torch.testing.assert_close(clock[0], -clock[1], atol=1e-6, rtol=0)

        env.mild_contact.foot_force[:, :, 2] = torch.tensor([[300.0, 0.0], [0.0, 300.0]])
        matching = walking_mdp.phase_contact(env)
        # Keeping the same support foot through the next half-cycle must lose
        # reward; swapping support restores it.
        env.episode_length_buf += 20
        wrong_foot = walking_mdp.phase_contact(env)
        self.assertTrue((matching > wrong_foot + 0.9).all())
        env.mild_contact.foot_force = env.mild_contact.foot_force.flip(1)
        torch.testing.assert_close(walking_mdp.phase_contact(env), matching)

    def test_turn_command_activates_clock_without_linear_motion(self):
        env = self.make_env(command=(0.0, 0.0, 0.5))
        torch.testing.assert_close(walking_mdp.gait_clock(env).norm(dim=-1), torch.ones(2))
        self.assertTrue((walking_mdp.desired_support(env).sum(dim=-1) == 1).all())

    def test_episode_reset_restarts_only_its_clock(self):
        env = self.make_env()
        before = walking_mdp.gait_clock(env).clone()
        env.episode_length_buf[0] = 0
        after = walking_mdp.gait_clock(env)
        torch.testing.assert_close(after[0], torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(after[1], before[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
