"""29-DoF G1 proprioceptive estimator ablations on the existing MILD task.

Only observation history and training labels differ between the ablations.
All share the alternating-gait clock and rewards with MILD walking, and the
project's full-body G1 actions, actuator gains, commands and contact physics.
"""

from dataclasses import dataclass
from typing import Literal

from mjlab.managers import ObservationTermCfg

from instinct_mj.tasks.locomotion.mild.vae_mdp import state_targets

from .mild_env_cfg import configure_mild_play
from .mild_gait_env_cfg import G1MildGaitEnvCfg

VaeVariant = Literal["baseline", "vae", "full"]


@dataclass(kw_only=True)
class G1MildVaeEnvCfg(G1MildGaitEnvCfg):
    history_length: int = 10
    use_vae: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.history_length < 1:
            raise ValueError("Estimator history_length must be positive")
        self.run_name = "G1MildVAE" if self.use_vae else "G1MildVAEBaseline"
        # MJLab owns the vectorized causal buffers and backfills only the
        # reset environments. Each term is flattened oldest -> newest.
        self.observations["policy"].history_length = self.history_length if self.use_vae else 0
        self.observations["policy"].flatten_history_dim = True
        critic = self.observations["critic"].terms
        if self.use_vae:
            # Keep the inherited critic, including true base velocity. The
            # extra labels are consumed only by the auxiliary loss.
            critic["state_targets"] = ObservationTermCfg(func=state_targets)
        else:
            critic.pop("state_targets", None)


def instinct_g1_mild_vae_env_cfg(*, play: bool = False, variant: VaeVariant = "full") -> G1MildVaeEnvCfg:
    if variant not in ("baseline", "vae", "full"):
        raise ValueError(f"Unknown VAE experiment variant: {variant}")
    cfg = G1MildVaeEnvCfg(use_vae=variant != "baseline")
    if play:
        configure_mild_play(cfg)
    return cfg
