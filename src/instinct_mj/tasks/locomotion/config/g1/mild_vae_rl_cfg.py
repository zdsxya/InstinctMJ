"""Task-local types exposed by instinct-train/play's CLI."""

from dataclasses import dataclass, field

from .mild_rl_cfg import G1MildPPORunnerCfg
from .rl_cfg import AlgorithmCfg, PolicyCfg


@dataclass(kw_only=True)
class MildVaePolicyCfg(PolicyCfg):
    class_name: str = "MildActorCritic"
    init_noise_std: float = 0.3
    history_length: int = 10
    latent_dim: int = 16
    encoder_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    decoder_hidden_dims: list[int] = field(default_factory=lambda: [128, 256])
    modulation_hidden_dims: list[int] = field(default_factory=lambda: [64, 64])


@dataclass(kw_only=True)
class MildVaeAlgorithmCfg(AlgorithmCfg):
    class_name: str = "MildVaePPO"
    regression_weight: float = 0.3
    vae_kl_weight: float = 0.001
    clip_min_std: float = 0.05


@dataclass(kw_only=True)
class G1MildVaeRunnerCfg(G1MildPPORunnerCfg):
    policy: MildVaePolicyCfg = field(default_factory=MildVaePolicyCfg)
    algorithm: MildVaeAlgorithmCfg = field(default_factory=MildVaeAlgorithmCfg)
    normalizers: dict = field(default_factory=dict)  # Fixed physical scaling in env.
    experiment_name: str = "g1_locomotion_mild_vae_change1"
    save_interval: int = 100
    log_interval: int = 1
    max_iterations: int = 20000
    wall_time_limit_s: float = 0.0
    convergence_min_updates: int = 1000
    convergence_window: int = 250
    convergence_relative_tolerance: float = 0.02
