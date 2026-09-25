"""AMP observations and UMR references for the existing MILD walking task."""

import os
from pathlib import Path

from mjlab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg, TerminationTermCfg

from instinct_mj.assets.unitree_g1 import (
    G1_MJCF_PATH,
    G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping,
    G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf,
)
from instinct_mj.envs import mdp
from instinct_mj.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg
from instinct_mj.motion_reference.motion_reference_cfg import MotionReferenceManagerCfg
from instinct_mj.motion_reference.utils import motion_interpolate_bilinear


def configure_walk_amp(cfg):
    """Add only the reference sensor and discriminator observations.

    Match Parkour's five observation terms, ten-frame history, and velocity
    scaling. Policy/critic observations, task rewards, and physics stay local
    to MILD. UMR_MOTION_DIR can relocate the validated dataset.
    """
    dataset = Path(os.environ.get(
        "UMR_MOTION_DIR", Path(__file__).resolve().parents[6] / "outputs/umr/motions"
    )).expanduser().resolve()
    reference = MotionReferenceManagerCfg(
        name="motion_reference",
        entity_name="robot",
        robot_model_path=G1_MJCF_PATH,
        link_of_interests=["left_ankle_roll_link", "right_ankle_roll_link"],
        symmetric_augmentation_link_mapping=[1, 0],
        symmetric_augmentation_joint_mapping=list(G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping),
        symmetric_augmentation_joint_reverse_buf=list(G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf),
        frame_interval_s=0.02,
        update_period=0.02,
        num_frames=10,
        mp_split_method="Even",
        motion_buffers={"walk_turn_stand": AmassMotionCfg(
            path=str(dataset),
            filtered_motion_selection_filepath=str(dataset / "motions.yaml"),
            motion_start_from_middle_range=(0.0, 0.9),
            ensure_link_below_zero_ground=False,
            motion_interpolate_func=motion_interpolate_bilinear,
            velocity_estimation_method="frontward",
        )},
    )
    cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != reference.name) + (reference,)
    cfg.events["match_motion_reference"] = EventTermCfg(func=mdp.match_motion_ref_with_scene, mode="startup")
    # As in Parkour, exhausted references are resampled without ending the
    # robot episode or replacing MILD's reset state with an expert pose.
    cfg.terminations["dataset_exhausted"] = TerminationTermCfg(
        func=mdp.dataset_exhausted, time_out=True,
        params={"reference_cfg": SceneEntityCfg("motion_reference"), "reset_without_notice": True},
    )
    policy_terms, reference_terms = {}, {}
    for name, policy_func, reference_func, scale in (
        ("projected_gravity", mdp.projected_gravity, mdp.projected_gravity_reference_as_state, 1.0),
        ("joint_pos_rel", mdp.joint_pos_rel, mdp.joint_pos_rel_reference_as_state, 1.0),
        ("joint_vel", mdp.joint_vel_rel, mdp.joint_vel_rel_reference_as_state, 0.05),
        ("base_lin_vel", mdp.base_lin_vel, mdp.base_lin_vel_reference_as_state, 1.0),
        ("base_ang_vel", mdp.base_ang_vel, mdp.base_ang_vel_reference_as_state, 1.0),
    ):
        policy_params = {"asset_cfg": SceneEntityCfg("robot", preserve_order=True)}
        reference_params = {"asset_cfg": SceneEntityCfg("motion_reference")}
        if name in ("joint_pos_rel", "joint_vel"):
            reference_params["robot_cfg"] = SceneEntityCfg("robot")
        policy_terms[name] = ObservationTermCfg(
            func=policy_func, params=policy_params, scale=scale,
            history_length=10, flatten_history_dim=True,
        )
        reference_terms[name] = ObservationTermCfg(
            func=reference_func, params=reference_params, scale=scale,
            history_length=10, flatten_history_dim=True,
        )
    for name, terms in (("amp_policy", policy_terms), ("amp_reference", reference_terms)):
        cfg.observations[name] = ObservationGroupCfg(
            terms=terms, concatenate_terms=False, enable_corruption=False,
        )
