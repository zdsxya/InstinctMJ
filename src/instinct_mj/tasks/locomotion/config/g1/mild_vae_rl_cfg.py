"""Comparable PPO, VAE and VAE+modulation configurations for 29-DoF G1."""

from dataclasses import dataclass, field

from .mild_rl_cfg import G1MildPPORunnerCfg
from .mild_vae_env_cfg import VaeVariant
from .rl_cfg import AlgorithmCfg, PolicyCfg


@dataclass(kw_only=True)
class MildVaePolicyCfg(PolicyCfg):
    class_name: str = "MildActorCritic"
    history_length: int = 10
    latent_dim: int = 16
    encoder_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    decoder_hidden_dims: list[int] = field(default_factory=lambda: [128, 256])
    modulation_hidden_dims: list[int] = field(default_factory=lambda: [64, 64])
    use_modulation: bool = False
    # Engineering scales: velocities in m/s, height errors relative to 0.1 m.
    # The state heads and supervised labels themselves remain in SI units.
    state_scales: tuple[float, float, float] = (1.0, 1.0, 10.0)


@dataclass(kw_only=True)
class MildVaeAlgorithmCfg(AlgorithmCfg):
    class_name: str = "MildVaePPO"
    regression_weight: float = 0.3
    vae_kl_weight: float = 0.001
    reconstruction_weight: float = 1.0
    base_velocity_weight: float = 1.0
    feet_velocity_weight: float = 1.0
    feet_height_weight: float = 1.0


@dataclass(kw_only=True)
class G1MildVaeBaselineRunnerCfg(G1MildPPORunnerCfg):
    # All three experiments use the same fixed observation convention, PPO
    # settings, rollout length, seed and training budget. In particular a
    # running critic normalizer must not change the supervised state labels.
    normalizers: dict = field(default_factory=dict)
    experiment_name: str = "g1_locomotion_mild_vae_baseline"
    save_interval: int = 100
    log_interval: int = 1


@dataclass(kw_only=True)
class G1MildVaeRunnerCfg(G1MildVaeBaselineRunnerCfg):
    policy: MildVaePolicyCfg = field(default_factory=MildVaePolicyCfg)
    algorithm: MildVaeAlgorithmCfg = field(default_factory=MildVaeAlgorithmCfg)
    experiment_name: str = "g1_locomotion_mild_vae"
    wall_time_limit_s: float = 0.0


@dataclass(kw_only=True)
class G1MildVaeFullRunnerCfg(G1MildVaeRunnerCfg):
    policy: MildVaePolicyCfg = field(default_factory=lambda: MildVaePolicyCfg(use_modulation=True))
    experiment_name: str = "g1_locomotion_mild_vae_full"


def g1_mild_vae_instinct_rl_cfg(variant: VaeVariant = "full") -> G1MildVaeBaselineRunnerCfg:
    configs = {
        "baseline": G1MildVaeBaselineRunnerCfg,
        "vae": G1MildVaeRunnerCfg,
        "full": G1MildVaeFullRunnerCfg,
    }
    if variant not in configs:
        raise ValueError(f"Unknown VAE experiment variant: {variant}")
    return configs[variant]()
