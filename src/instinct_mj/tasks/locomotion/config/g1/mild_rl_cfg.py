"""Ordinary PPO for the contact ablation; MLP plus AMP for MILD walking."""

from dataclasses import dataclass, field

from .rl_cfg import AlgorithmCfg, G1FlatPPORunnerCfg


@dataclass(kw_only=True)
class G1MildPPORunnerCfg(G1FlatPPORunnerCfg):
    # The flat config annotates this as object, which tyro exposes as {fixed}.
    # A concrete CLI type makes checkpoint resume usable without changing PPO.
    load_run: str | None = None
    experiment_name: str = "g1_locomotion_mild_contact"


@dataclass(kw_only=True)
class MildWalkAmpAlgorithmCfg(AlgorithmCfg):
    """Parkour discriminator with a lower style weight for upright walking."""

    class_name: str = "WasabiPPO"
    discriminator_kwargs: dict = field(default_factory=lambda: {
        "hidden_sizes": [1024, 512], "nonlinearity": "ReLU",
    })
    # AMP is added per step, while environment rewards are scaled by dt=0.02.
    # Reduce style pressure so the explicit posture penalties have more influence.
    discriminator_reward_coef: float = 0.025
    discriminator_reward_type: str = "quad"
    discriminator_loss_func: str = "MSELoss"
    discriminator_gradient_penalty_coef: float = 5.0
    discriminator_optimizer_class_name: str = "AdamW"
    discriminator_weight_decay_coef: float = 3e-4
    discriminator_logit_weight_decay_coef: float = 0.04
    discriminator_optimizer_kwargs: dict = field(default_factory=lambda: {
        "lr": 1e-4, "betas": [0.9, 0.999],
    })


@dataclass(kw_only=True)
class G1MildWalkPPORunnerCfg(G1MildPPORunnerCfg):
    algorithm: MildWalkAmpAlgorithmCfg = field(default_factory=MildWalkAmpAlgorithmCfg)
    experiment_name: str = "g1_locomotion_mild_walk_amp"


def g1_mild_walk_instinct_rl_cfg() -> G1MildWalkPPORunnerCfg:
    return G1MildWalkPPORunnerCfg()
