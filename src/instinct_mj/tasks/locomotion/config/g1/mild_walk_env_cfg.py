"""Forward alternating walking with UMR AMP, separate from the contact ablation."""

import math
from dataclasses import dataclass

from mjlab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg

from instinct_mj.tasks.locomotion.mild import walking_mdp
from instinct_mj.tasks.locomotion.mild.arm_reference import ReferenceShoulderPitchPenalty, ReferenceShoulderRollPenalty

from .mild_env_cfg import G1MildEnvCfg, configure_mild_play
from .mild_walk_amp_cfg import configure_walk_amp


@dataclass(kw_only=True)
class G1MildWalkEnvCfg(G1MildEnvCfg):
    gait_period: float = 0.8
    gait_stance_fraction: float = 0.6
    gait_clearance: float = 0.065
    gait_contact_threshold: float = 5.0

    def __post_init__(self):
        super().__post_init__()
        if self.gait_period <= 0 or not 0.5 < self.gait_stance_fraction < 1:
            raise ValueError("Walking requires a positive period and stance fraction between 0.5 and 1")
        if self.gait_clearance <= 0 or self.gait_contact_threshold <= 0:
            raise ValueError("Walking clearance and contact threshold must be positive")
        self.run_name = "G1MildWalkAMP"
        for group in ("policy", "critic"):
            self.observations[group].terms["gait_clock"] = ObservationTermCfg(func=walking_mdp.gait_clock)
        self.rewards["gait_contact"] = RewardTermCfg(func=walking_mdp.phase_contact, weight=2.0)
        self.rewards["swing_clearance"] = RewardTermCfg(func=walking_mdp.swing_clearance, weight=0.6)
        self.rewards["both_feet_airborne"] = RewardTermCfg(func=walking_mdp.flight, weight=-2.0)
        self.rewards["lin_vel_z_l2"].weight = -0.5
        # The free root is torso_link: penalize chest tilt and waist bending
        # separately. Keep these overrides local to the AMP walking task.
        self.rewards["flat_orientation_l2"].weight = -3.0
        self.rewards["joint_deviation_torso"].weight = -0.5
        # Bias shoulder, elbow and wrist joints toward the default pose in
        # addition to AMP and the reference-derived shoulder limits below.
        self.rewards["joint_deviation_arms"].weight = -0.1
        # Arms use the dedicated posture penalty above during both standing
        # and walking; apply the standing-specific penalty to waist and legs.
        self.rewards["stand_still"].params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=("waist_.*", ".*_hip_.*", ".*_knee_joint", ".*_ankle_.*")
        )
        # The broad inherited yaw kernel barely distinguishes a slow drift
        # from zero turn rate. Increase precision and retain a cost near zero.
        self.rewards["track_ang_vel_z_exp"].weight = 2.0
        self.rewards["track_ang_vel_z_exp"].params["std"] = 0.25
        self.rewards["yaw_rate_error_l1"] = RewardTermCfg(func=walking_mdp.yaw_rate_error_l1, weight=-0.5)
        # Add penalties for excessive lateral motion and forward raise beyond
        # the reference envelope, alongside the default-pose penalty above.
        self.rewards["arm_reference_roll"] = RewardTermCfg(
            func=ReferenceShoulderRollPenalty, weight=-0.5,
            params={"reference_name": "motion_reference", "buffer_name": "walk_turn_stand"},
        )
        self.rewards["arm_reference_pitch"] = RewardTermCfg(
            func=ReferenceShoulderPitchPenalty, weight=-0.5,
            params={
                "reference_name": "motion_reference", "buffer_name": "walk_turn_stand",
                "quantile": 0.01, "position_margin": math.radians(2),
            },
        )
        # First establish forward walking; turning/sideways robustness is a later stage.
        command = self.commands["base_velocity"]
        command.heading_command = False
        command.ranges.heading = None
        command.rel_standing_envs = 0.1
        command.ranges.lin_vel_x = (0.2, 0.6)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        self.events["push_robot"] = None
        configure_walk_amp(self)


def instinct_g1_mild_walk_env_cfg(*, play=False):
    cfg = G1MildWalkEnvCfg()
    if play:
        configure_mild_play(cfg)
    return cfg
