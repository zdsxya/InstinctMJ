"""Walking arm posture, reference limits and standing reward selection."""

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from mjlab.managers import SceneEntityCfg

from instinct_mj.tasks.locomotion.mdp.rewards import joint_deviation_l1, stand_still
from instinct_mj.tasks.registry import load_env_cfg
from instinct_mj.tasks.locomotion.mild.arm_reference import (
    ReferenceShoulderPitchPenalty,
    ReferenceShoulderRollPenalty,
    load_shoulder_pitch_envelope,
    load_shoulder_roll_envelope,
    shoulder_pitch_excess,
    shoulder_roll_excess,
)
from instinct_mj.tasks.locomotion.mild.walking_mdp import yaw_rate_error_l1


class ArmRewardTests(unittest.TestCase):
    def test_standing_selection_and_command_gate(self):
        # Two lower-body joints and two arm joints, with the arms far from rest.
        q = torch.tensor([[.2, -.3, 1., -2.]]).repeat(3, 1)
        commands = torch.tensor([[0., 0., 0.], [.2, 0., 0.], [0., 0., .2]])
        robot = SimpleNamespace(data=SimpleNamespace(joint_pos=q, default_joint_pos=torch.zeros_like(q)))
        env = SimpleNamespace(scene={"robot": robot}, command_manager=SimpleNamespace(get_command=lambda _: commands))
        selected = SceneEntityCfg("robot", joint_ids=[0, 1])
        torch.testing.assert_close(stand_still(env, "velocity", selected), torch.tensor([.5, 0., 0.]))
        q[:, 2:] *= 10
        torch.testing.assert_close(stand_still(env, "velocity", selected), torch.tensor([.5, 0., 0.]))
        # The default selection preserves the original all-joints behavior.
        torch.testing.assert_close(stand_still(env, "velocity"), torch.tensor([30.5, 0., 0.]))

    def test_arm_posture_and_reference_limits_in_walk_and_play(self):
        from instinct_mj.tasks.locomotion.config.g1.mild_env_cfg import G1MildEnvCfg

        mild = G1MildEnvCfg()
        for suffix in ("v0", "Play-v0"):
            walk = load_env_cfg(f"Instinct-Locomotion-Mild-Walk-G1-{suffix}", play=suffix == "Play-v0")
            posture = walk.rewards["joint_deviation_arms"]
            self.assertIs(posture.func, joint_deviation_l1)
            self.assertEqual(posture.weight, -.1)
            self.assertEqual(posture.params["asset_cfg"].joint_names,
                             mild.rewards["joint_deviation_arms"].params["asset_cfg"].joint_names)
            self.assertEqual(walk.rewards["stand_still"].params["asset_cfg"].joint_names,
                             ("waist_.*", ".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"))
            self.assertIs(walk.rewards["arm_reference_roll"].func, ReferenceShoulderRollPenalty)
            self.assertEqual(walk.rewards["arm_reference_roll"].weight, -.5)
            self.assertIs(walk.rewards["arm_reference_pitch"].func, ReferenceShoulderPitchPenalty)
            self.assertEqual(walk.rewards["arm_reference_pitch"].weight, -.5)
            # AMP still observes all joint positions and velocities.
            for group in ("amp_policy", "amp_reference"):
                for term in ("joint_pos_rel", "joint_vel"):
                    self.assertIsNone(walk.observations[group].terms[term].params["asset_cfg"].joint_names)
        self.assertEqual(mild.rewards["joint_deviation_arms"].weight, -.1)
        self.assertNotIn("asset_cfg", mild.rewards["stand_still"].params)
        self.assertNotIn("arm_reference_roll", mild.rewards)
        self.assertNotIn("arm_reference_pitch", mild.rewards)
        self.assertNotIn("yaw_rate_error_l1", mild.rewards)
        self.assertEqual(mild.rewards["track_ang_vel_z_exp"].weight, 1.)
        from instinct_mj.tasks.locomotion.config.g1.mild_vae_env_cfg import instinct_g1_mild_vae_env_cfg

        for cfg in (load_env_cfg("Instinct-Locomotion-Flat-G1-v0"), instinct_g1_mild_vae_env_cfg()):
            self.assertNotIn("arm_reference_pitch", cfg.rewards)

    def test_reference_band_has_no_default_pose_target_and_is_mirror_symmetric(self):
        position = torch.tensor([[.2, -.3], [.1, -.4], [.2, -.3]])
        velocity = torch.tensor([[.1, -.2], [.1, -.2], [.8, -.8]])
        cost = shoulder_roll_excess(position, velocity, .15, .35, .5)
        self.assertEqual(cost[0].item(), 0)
        self.assertGreater(cost[1].item(), 0)
        self.assertGreater(cost[2].item(), 0)
        torch.testing.assert_close(cost, shoulder_roll_excess(-position.flip(-1), -velocity.flip(-1), .15, .35, .5))

    def test_envelope_uses_selected_files_and_no_interclip_velocity(self):
        with TemporaryDirectory() as directory:
            p = Path(directory)
            for name, angle in (("a", .2), ("b", .4), ("excluded", 9.)):
                # Deliberately reverse the stored joint order.
                np.savez(p / (name + '.npz'), joint_names=['right_shoulder_roll_joint', 'left_shoulder_roll_joint'],
                         joint_pos=np.tile([-angle, angle], (20, 1)), framerate=50.)
            (p / 'selection.yaml').write_text('selected_files: [a.npz, b.npz]\n')
            stats = load_shoulder_roll_envelope(p, p / 'selection.yaml', quantile=0, position_margin=0, velocity_margin=0)
            self.assertAlmostEqual(stats['lower_rad'], .2)
            self.assertAlmostEqual(stats['upper_rad'], .4)
            self.assertEqual(stats['speed_rad_s'], 0)
            self.assertEqual(len(stats['sources']), 2)

    def test_pitch_penalty_only_limits_forward_raise_and_is_mirror_symmetric(self):
        position = torch.deg2rad(torch.tensor([[-15., -10.], [-20., -25.], [0., 30.]]))
        cost = shoulder_pitch_excess(position, math.radians(-15))
        # At the boundary, inside the reference band, and backward swing are free.
        # Exceeding the limit by 5 and 10 degrees gives mean squared cost 2.5.
        torch.testing.assert_close(cost, torch.tensor([0., 2.5, 0.]))
        # Pitch uses the same sign on both sides, unlike shoulder roll.
        torch.testing.assert_close(cost, shoulder_pitch_excess(position.flip(-1), math.radians(-15)))

    def test_pitch_reference_and_runtime_joint_mapping(self):
        with TemporaryDirectory() as directory:
            p = Path(directory)
            for name, angle in (("a", -.2), ("b", -.4), ("excluded", -9.)):
                np.savez(p / (name + '.npz'),
                         joint_names=['right_shoulder_pitch_joint', 'left_elbow_joint', 'left_shoulder_pitch_joint'],
                         joint_pos=np.tile([angle, -20., angle + .1], (20, 1)))
            selection = p / 'selection.yaml'
            selection.write_text('selected_files: [a.npz, b.npz]\n')
            stats = load_shoulder_pitch_envelope(p, selection, quantile=0, position_margin=.03)
            self.assertAlmostEqual(stats['lower_rad'], -.43)
            self.assertEqual(len(stats['sources']), 2)
            self.assertTrue(all(len(source['sha256']) == 64 for source in stats['sources']))

            buffer_cfg = SimpleNamespace(path=p, filtered_motion_selection_filepath=selection, subset_selection=None)
            reference = SimpleNamespace(cfg=SimpleNamespace(motion_buffers={'motions': buffer_cfg}))
            q = torch.tensor([[-30., -.4, 40., -.2], [30., -.5, -40., -.6]])
            robot = SimpleNamespace(
                joint_names=['left_elbow_joint', 'right_shoulder_pitch_joint',
                             'left_shoulder_roll_joint', 'left_shoulder_pitch_joint'],
                data=SimpleNamespace(joint_pos=q),
            )
            env = SimpleNamespace(scene={'robot': robot, 'reference': reference})
            cfg = SimpleNamespace(params={
                'reference_name': 'reference', 'buffer_name': 'motions',
                'quantile': 0., 'position_margin': .03,
            })
            penalty = ReferenceShoulderPitchPenalty(cfg, env)
            expected = torch.tensor([0., ((.07 / math.radians(5))**2 + (.17 / math.radians(5))**2) / 2])
            torch.testing.assert_close(penalty(env, **cfg.params), expected)

    def test_negative_pitch_raises_both_g1_arms_forward(self):
        import mujoco
        from instinct_mj.assets.unitree_g1 import G1_MJCF_PATH

        model = mujoco.MjModel.from_xml_path(G1_MJCF_PATH)
        data = mujoco.MjData(model)
        for side in ('left', 'right'):
            joint = model.joint(f'{side}_shoulder_pitch_joint')
            forward_distances = []
            for pitch in (.2, -.2):
                data.qpos[joint.qposadr] = pitch
                mujoco.mj_kinematics(model, data)
                upper = data.body(f'{side}_elbow_link').xpos - data.body(f'{side}_shoulder_roll_link').xpos
                torso_rotation = data.body('torso_link').xmat.reshape(3, 3)
                forward_distances.append((upper @ torso_rotation)[0])
            self.assertGreater(forward_distances[1], forward_distances[0])

    def test_pitch_envelope_rejects_invalid_reference_data(self):
        with TemporaryDirectory() as directory:
            p = Path(directory)
            selection = p / 'selection.yaml'
            selection.write_text('selected_files: []\n')
            with self.assertRaises(ValueError):
                load_shoulder_pitch_envelope(p, selection)
            selection.write_text('selected_files: [bad.npz]\n')
            np.savez(p / 'bad.npz', joint_names=['left_shoulder_pitch_joint', 'right_shoulder_pitch_joint'],
                     joint_pos=np.array([[float('nan'), 0.]]))
            with self.assertRaises(ValueError):
                load_shoulder_pitch_envelope(p, selection)

    def test_yaw_penalty_tracks_command_with_both_signs(self):
        commands = torch.tensor([[0., 0., 0.], [0., 0., -.2], [0., 0., .2]])
        velocity = torch.tensor([[0., 0., -.05], [0., 0., -.2], [0., 0., .25]])
        env = SimpleNamespace(scene={'robot': SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_w=velocity))},
                              command_manager=SimpleNamespace(get_command=lambda _: commands))
        torch.testing.assert_close(yaw_rate_error_l1(env), torch.tensor([.05, 0., .05]))


if __name__ == "__main__":
    unittest.main()
