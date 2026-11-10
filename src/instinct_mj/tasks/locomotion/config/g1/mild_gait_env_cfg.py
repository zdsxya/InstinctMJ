"""Shared alternating-gait observations and rewards for MILD locomotion."""

import math
from dataclasses import dataclass

from mjlab.managers import ObservationTermCfg, RewardTermCfg

from instinct_mj.tasks.locomotion.mild import walking_mdp

from .mild_env_cfg import G1MildEnvCfg


@dataclass(kw_only=True)
class G1MildGaitEnvCfg(G1MildEnvCfg):
    gait_period: float = 0.8
    gait_stance_fraction: float = 0.6
    gait_clearance: float = 0.065
    gait_contact_threshold: float = 5.0
    randomize_gait_phase_by_env: bool = True

    def __post_init__(self):
        super().__post_init__()
        for name in ("gait_period", "gait_clearance", "gait_contact_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.5 < self.gait_stance_fraction < 1:
            raise ValueError("gait_stance_fraction must be between 0.5 and 1")
        if not isinstance(self.randomize_gait_phase_by_env, bool):
            raise ValueError("randomize_gait_phase_by_env must be a bool")
        for group in ("policy", "critic"):
            self.observations[group].terms["gait_clock"] = ObservationTermCfg(func=walking_mdp.gait_clock)
        self.rewards["gait_contact"] = RewardTermCfg(func=walking_mdp.phase_contact, weight=2.0)
        self.rewards["swing_clearance"] = RewardTermCfg(func=walking_mdp.swing_clearance, weight=0.6)
        self.rewards["both_feet_airborne"] = RewardTermCfg(func=walking_mdp.flight, weight=-2.0)
