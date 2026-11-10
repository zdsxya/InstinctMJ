"""Reusable proprioceptive state estimation and latent modulation networks."""

import math

import torch
from instinct_rl.modules.actor_critic import get_activation
from torch import nn


def mlp(input_dim, hidden_dims, output_dim, activation):
    layers = []
    for width in hidden_dims:
        if width < 1:
            raise ValueError("MLP hidden dimensions must be positive")
        layers.extend((nn.Linear(input_dim, width), get_activation(activation)))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class ProprioceptiveVAE(nn.Module):
    """Encode causal observation history; predict physical states and a latent.

    State heads produce SI quantities: base velocity, two foot velocities, and
    two foot heights. Only the reconstruction branch samples the latent; the
    policy uses its mean so PPO can reproduce rollout action probabilities.
    """

    def __init__(
        self,
        history_dim,
        frame_dim,
        latent_dim=16,
        encoder_hidden_dims=(256, 128),
        decoder_hidden_dims=(128, 256),
        activation="elu",
    ):
        super().__init__()
        if min(history_dim, frame_dim, latent_dim) < 1 or not encoder_hidden_dims:
            raise ValueError("Estimator dimensions and encoder hidden layers must be nonempty")
        self.history_dim, self.frame_dim, self.latent_dim = history_dim, frame_dim, latent_dim
        width = encoder_hidden_dims[-1]
        self.encoder = mlp(history_dim, encoder_hidden_dims[:-1], width, activation)
        self.encoder.add_module("activation", get_activation(activation))
        self.state_heads = nn.ModuleDict(
            {
                "base_velocity": nn.Linear(width, 3),
                "feet_velocity": nn.Linear(width, 6),
                "feet_height": nn.Linear(width, 2),
            }
        )
        self.latent_mean = nn.Linear(width, latent_dim)
        self.latent_logvar = nn.Linear(width, latent_dim)
        self.decoder = mlp(latent_dim, decoder_hidden_dims, frame_dim, activation)

    @property
    def base_velocity_head(self):
        return self.state_heads["base_velocity"]

    @property
    def feet_velocity_head(self):
        return self.state_heads["feet_velocity"]

    @property
    def feet_height_head(self):
        return self.state_heads["feet_height"]

    def forward(self, history):
        features = self.encoder(history)
        state = torch.cat([head(features) for head in self.state_heads.values()], dim=-1)
        return state, self.latent_mean(features), self.latent_logvar(features).clamp(-10, 6)

    @staticmethod
    def reparameterize(mean, logvar):
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)


class FootVelocityModulation(nn.Module):
    """Residual affine modulation driven only by estimated foot velocities."""

    def __init__(self, latent_dim=16, hidden_dims=(64, 64), activation="elu"):
        super().__init__()
        if not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("Modulation latent dimension must be a positive integer")
        self.network = mlp(6, hidden_dims, 2 * latent_dim, activation)
        # Near identity while retaining gradients in every hidden layer.
        nn.init.normal_(self.network[-1].weight, std=0.001)
        nn.init.zeros_(self.network[-1].bias)

    def parameters_for(self, feet_velocity):
        return self.network(feet_velocity).chunk(2, dim=-1)

    def forward(self, feet_velocity, latent):
        scale, shift = self.parameters_for(feet_velocity)
        return (1 + scale) * latent + shift


def state_scale_vector(scales):
    """Expand configurable base/foot velocity and height scales to 11 values."""
    if len(scales) != 3 or any(not math.isfinite(value) or value <= 0 for value in scales):
        raise ValueError("State scales must contain three finite positive values")
    return torch.tensor([scales[0]] * 3 + [scales[1]] * 6 + [scales[2]] * 2, dtype=torch.float32)
