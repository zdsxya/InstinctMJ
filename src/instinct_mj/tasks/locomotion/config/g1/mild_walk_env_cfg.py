"""Forward alternating walking with UMR AMP, separate from the contact ablation."""

from dataclasses import dataclass

from mjlab.managers import ObservationTermCfg, RewardTermCfg

from instinct_mj.tasks.locomotion.mild import walking_mdp

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
