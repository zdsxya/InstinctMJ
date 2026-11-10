"""Gait-conditioned VAE with supervised heads and foot-velocity modulation."""

import math
from pathlib import Path

import torch
from instinct_rl.modules.actor_critic import ActorCritic
from torch import nn
from torch.distributions import Normal

from .estimator import FootVelocityModulation, ProprioceptiveVAE, mlp, state_scale_vector


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
        use_modulation=True,
        state_scales=(1.0, 1.0, 10.0),
        **kwargs,
    ):
        nn.Module.__init__(self)
        if kwargs:
            raise TypeError(f"Unknown MILD policy options: {sorted(kwargs)}")
        if not isinstance(history_length, int) or history_length < 1:
            raise ValueError("History length must be a positive integer")
        if not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("Latent dimension must be a positive integer")
        self.format = obs_format
        self.history_length, self.latent_dim = history_length, latent_dim
        self.use_modulation = use_modulation
        self.policy_slices = segments(obs_format["policy"])
        self.critic_slices = segments(obs_format["critic"])
        self.frame_dims = []
        for shape in obs_format["policy"].values():
            if math.prod(shape) < history_length or math.prod(shape) % history_length:
                raise ValueError("Policy term dimensions must be divisible by history_length")
            self.frame_dims.append(math.prod(shape) // history_length)
        self.frame_dim = sum(self.frame_dims)
        self.history_dim = self.frame_dim * history_length
        if set(self.policy_slices) != {
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
            "gait_clock",
        }:
            raise ValueError("MILD encoder requires the six proprioceptive terms and gait_clock")
        if math.prod(obs_format["policy"]["gait_clock"]) // history_length != 2:
            raise ValueError("MILD gait_clock must contain sine and cosine per frame")
        if math.prod(obs_format["policy"]["actions"]) // history_length != num_actions:
            raise ValueError("Previous-action observation must match the action dimension")
        if self.critic_slices["state_targets"].stop - self.critic_slices["state_targets"].start != 11:
            raise ValueError("MILD requires 11 state targets")
        self.estimator = ProprioceptiveVAE(
            self.history_dim, self.frame_dim, latent_dim, encoder_hidden_dims, decoder_hidden_dims, activation
        )
        self.modulation = (
            FootVelocityModulation(latent_dim, modulation_hidden_dims, activation) if use_modulation else None
        )
        self.register_buffer("state_scales", state_scale_vector(state_scales))
        self.actor = mlp(self.frame_dim + 11 + latent_dim, actor_hidden_dims, num_actions, activation)
        # Supervision is carried alongside critic data in rollout storage, but
        # never changes the baseline critic's network input. Old history fields
        # are excluded too, rather than silently introducing an estimator path.
        self.critic_input_slices = tuple(
            section for name, section in self.critic_slices.items() if name not in {"state_targets", "policy_history"}
        )
        self.privileged_dim = sum(section.stop - section.start for section in self.critic_input_slices)
        if not self.privileged_dim:
            raise ValueError("At least one original critic observation is required")
        self.critic = mlp(self.privileged_dim, critic_hidden_dims, num_rewards, activation)
        self.std = nn.Parameter(torch.full((num_actions,), float(init_noise_std)))
        self.distribution = None

    @property
    def encoder(self):
        return self.estimator.encoder

    @property
    def state_head(self):
        return self.estimator.state_heads

    @property
    def latent_mean(self):
        return self.estimator.latent_mean

    @property
    def latent_logvar(self):
        return self.estimator.latent_logvar

    @property
    def decoder(self):
        return self.estimator.decoder

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
        return self.estimator(history)

    def modulate(self, estimate, latent):
        if self.modulation is None:
            return latent
        return self.modulation(estimate[..., 3:9], latent)

    def act_inference(self, observations):
        estimate, mean, _ = self.encode(observations)
        return self.actor(
            torch.cat(
                (self.current_frame(observations), estimate * self.state_scales, self.modulate(estimate, mean)), dim=-1
            )
        )

    def update_distribution(self, observations):
        mean = self.act_inference(observations)
        self.distribution = Normal(mean, self.std.clamp(min=1e-4).expand_as(mean))

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(torch.cat([critic_observations[..., section] for section in self.critic_input_slices], dim=-1))

    def auxiliary_losses(self, history, critic_observations):
        return self.auxiliary_loss_and_statistics(history, critic_observations)[0]

    def auxiliary_loss_and_statistics(self, history, critic_observations):
        estimate, mean, logvar = self.encode(history)
        sample = self.estimator.reparameterize(mean, logvar)
        targets = critic_observations[..., self.critic_slices["state_targets"]]
        physical_error = estimate - targets
        error = physical_error * self.state_scales
        losses = {
            "reconstruction": (self.decoder(sample) - self.current_frame(history)).square().mean(),
            "vae_kl": -0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(-1).mean(),
            "base_velocity": error[..., :3].square().mean(),
            "feet_velocity": error[..., 3:9].square().mean(),
            "feet_height": error[..., 9:].square().mean(),
        }
        with torch.no_grad():
            if self.modulation is None:
                scale = shift = mean.new_zeros(())
            else:
                scale, shift = self.modulation.parameters_for(estimate[..., 3:9])
            statistics = {
                # Report physical squared units while optimizing the scaled
                # losses above; height scaling must not obscure evaluation.
                "VAE/Base_Velocity_MSE": physical_error[..., :3].square().mean(),
                "VAE/Foot_Velocity_MSE": physical_error[..., 3:9].square().mean(),
                "VAE/Foot_Height_MSE": physical_error[..., 9:].square().mean(),
                "VAE/Latent_Mean": mean.mean(),
                "VAE/Latent_Std": torch.exp(0.5 * logvar).mean(),
                "VAE/Modulation_Scale": scale.mean(),
                "VAE/Modulation_Shift": shift.mean(),
            }
        return losses, statistics

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
