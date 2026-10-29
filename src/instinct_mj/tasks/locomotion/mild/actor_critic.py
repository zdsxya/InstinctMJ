"""Fig. 4: proprioceptive VAE with supervised heads and foot-velocity FiLM."""

import math
from pathlib import Path

import torch
from instinct_rl.modules.actor_critic import ActorCritic, get_activation
from torch import nn
from torch.distributions import Normal


def mlp(input_dim, hidden, output_dim, activation):
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(input_dim, width), get_activation(activation)))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


def segments(format):
    result, offset = {}, 0
    for name, shape in format.items():
        size = math.prod(shape)
        result[name] = slice(offset, offset + size)
        offset += size
    return result


class MildActorCritic(ActorCritic):
    def __init__(
        self,
        obs_format,
        num_actions,
        num_rewards=1,
        history_length=10,
        latent_dim=16,
        encoder_hidden_dims=(256, 128),
        decoder_hidden_dims=(128, 256),
        modulation_hidden_dims=(64, 64),
        actor_hidden_dims=(256, 128, 128),
        critic_hidden_dims=(256, 128, 128),
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        nn.Module.__init__(self)
        if kwargs:
            raise TypeError(f"Unknown MILD policy options: {sorted(kwargs)}")
        self.format = obs_format
        self.history_length, self.latent_dim = history_length, latent_dim
        self.policy_slices = segments(obs_format["policy"])
        self.critic_slices = segments(obs_format["critic"])
        self.frame_dims = []
        for shape in obs_format["policy"].values():
            if math.prod(shape) % history_length:
                raise ValueError("Policy term dimensions must be divisible by history_length")
            self.frame_dims.append(math.prod(shape) // history_length)
        self.frame_dim = sum(self.frame_dims)
        self.history_dim = self.frame_dim * history_length
        if self.frame_dim != 9 + 3 * num_actions:
            raise ValueError("MILD expects 9 + 3 * num_actions proprioceptive values")
        if list(self.policy_slices) != [
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
        ]:
            raise ValueError("MILD policy schema/order changed")
        if self.critic_slices["state_targets"].stop - self.critic_slices["state_targets"].start != 11:
            raise ValueError("MILD requires 11 state targets")
        if math.prod(obs_format["critic"]["policy_history"]) != self.history_dim:
            raise ValueError("Critic must store the identical actor history")
        self.encoder = mlp(self.history_dim, encoder_hidden_dims[:-1], encoder_hidden_dims[-1], activation)
        self.encoder.add_module("activation", get_activation(activation))
        width = encoder_hidden_dims[-1]
        self.state_head = nn.Linear(width, 11)
        self.latent_mean = nn.Linear(width, latent_dim)
        self.latent_logvar = nn.Linear(width, latent_dim)
        self.decoder = mlp(latent_dim, decoder_hidden_dims, self.frame_dim, activation)
        self.modulation = mlp(6, modulation_hidden_dims, 2 * latent_dim, activation)
        # Start at identity plus a small perturbation, preserving nonzero
        # gradients throughout the modulation MLP on the first update.
        nn.init.normal_(self.modulation[-1].weight, std=0.001)
        nn.init.zeros_(self.modulation[-1].bias)
        self.actor = mlp(self.frame_dim + 11 + latent_dim, actor_hidden_dims, num_actions, activation)
        self.privileged_dim = self.critic_slices["policy_history"].start
        self.critic = mlp(self.privileged_dim + latent_dim, critic_hidden_dims, num_rewards, activation)
        self.std = nn.Parameter(torch.full((num_actions,), float(init_noise_std)))
        self.distribution = None

    @property
    def obs_segments(self):
        return self.format["policy"]

    @property
    def critic_obs_segments(self):
        return self.format["critic"]

    def current_frame(self, history):
        return torch.cat(
            [
                history[..., section].reshape(*history.shape[:-1], self.history_length, width)[..., -1, :]
                for section, width in zip(self.policy_slices.values(), self.frame_dims, strict=True)
            ],
            dim=-1,
        )

    def encode(self, history):
        features = self.encoder(history)
        return self.state_head(features), self.latent_mean(features), self.latent_logvar(features).clamp(-10, 6)

    def modulate(self, estimate, latent):
        scale_delta, shift = self.modulation(estimate[..., 3:9]).chunk(2, dim=-1)
        return (1 + scale_delta) * latent + shift

    def act_inference(self, observations):
        estimate, mean, _ = self.encode(observations)
        return self.actor(
            torch.cat((self.current_frame(observations), estimate, self.modulate(estimate, mean)), dim=-1)
        )

    def update_distribution(self, observations):
        mean = self.act_inference(observations)
        self.distribution = Normal(mean, self.std.clamp(min=1e-4).expand_as(mean))

    def evaluate(self, critic_observations, **kwargs):
        history = critic_observations[..., self.critic_slices["policy_history"]]
        estimate, mean, _ = self.encode(history)
        return self.critic(
            torch.cat((critic_observations[..., : self.privileged_dim], self.modulate(estimate, mean)), dim=-1)
        )

    def auxiliary_losses(self, history, critic_observations):
        estimate, mean, logvar = self.encode(history)
        sample = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        targets = critic_observations[..., self.critic_slices["state_targets"]]
        return {
            "reconstruction": (self.decoder(sample) - self.current_frame(history)).square().mean(),
            "vae_kl": -0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(-1).mean(),
            "base_velocity": (estimate[..., :3] - targets[..., :3]).square().mean(),
            "feet_velocity": (estimate[..., 3:9] - targets[..., 3:9]).square().mean(),
            "feet_height": (estimate[..., 9:] - targets[..., 9:]).square().mean(),
        }

    def forward(self, observations):
        return self.act_inference(observations)

    def export_as_onnx(self, observations, filedir):
        self.eval()
        Path(filedir).mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            self,
            observations[:1],
            str(Path(filedir) / "actor.onnx"),
            input_names=["policy_history"],
            output_names=["actions"],
            dynamic_axes={"policy_history": {0: "batch"}, "actions": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
