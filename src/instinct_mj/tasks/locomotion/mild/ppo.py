"""Joint PPO and proprioceptive-estimator optimization for the G1 ablations."""

import math

import torch
import torch.distributed as dist
from instinct_rl.algorithms.ppo import PPO


class MildVaePPO(PPO):
    def __init__(
        self,
        *args,
        regression_weight=0.3,
        vae_kl_weight=0.001,
        reconstruction_weight=1.0,
        base_velocity_weight=1.0,
        feet_velocity_weight=1.0,
        feet_height_weight=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.auxiliary_weights = {
            "reconstruction": reconstruction_weight,
            "vae_kl": vae_kl_weight,
            "base_velocity": base_velocity_weight,
            "feet_velocity": feet_velocity_weight,
            "feet_height": feet_height_weight,
        }
        if not 0 < regression_weight < 1 or any(
            not math.isfinite(value) or value < 0 for value in self.auxiliary_weights.values()
        ):
            raise ValueError("Invalid MILD joint loss weights")
        self.regression_weight, self.vae_kl_weight = regression_weight, vae_kl_weight
        # PPO.update multiplies these exactly once.
        self.surrogate_loss_coef = 1 - regression_weight
        self.value_loss_coef *= 1 - regression_weight
        self.entropy_coef *= 1 - regression_weight
        self.regression_loss_coef = regression_weight

    def compute_losses(self, minibatch):
        losses, intermediate, stats = super().compute_losses(minibatch)
        auxiliary, diagnostics = self.actor_critic.auxiliary_loss_and_statistics(minibatch.obs, minibatch.critic_obs)
        losses["regression_loss"] = sum(value * self.auxiliary_weights[name] for name, value in auxiliary.items())
        stats.update({f"estimator/{name}": value.detach() for name, value in auxiliary.items()})
        stats.update(diagnostics)
        stats.update(
            {
                "Train/PPO_Actor_Loss": losses["surrogate_loss"].detach(),
                "Train/PPO_Critic_Loss": losses["value_loss"].detach(),
                "Train/Policy_Entropy": -losses["entropy"].detach(),
                "VAE/Reconstruction_Loss": auxiliary["reconstruction"].detach(),
                "VAE/KL_Loss": auxiliary["vae_kl"].detach(),
                "VAE/Base_Velocity_Scaled_MSE": auxiliary["base_velocity"].detach(),
                "VAE/Foot_Velocity_Scaled_MSE": auxiliary["feet_velocity"].detach(),
                "VAE/Foot_Height_Scaled_MSE": auxiliary["feet_height"].detach(),
            }
        )
        stats["ppo_ratio_deviation"] = (intermediate["ratio"].detach() - 1).abs().mean()
        if "kl" in intermediate:
            stats["ppo_kl"] = intermediate["kl"].detach().mean()
        return losses, intermediate, stats

    def gradient_step(self, loss, average_stats):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite MILD joint loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if dist.is_initialized():
            for parameter in self.actor_critic.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                    parameter.grad /= dist.get_world_size()
        norm = torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm, error_if_nonfinite=True)
        average_stats["grad_norm"] += norm.detach()
        for name in ("encoder", "state_head", "latent_mean", "latent_logvar", "decoder", "modulation", "actor", "critic"):
            module = getattr(self.actor_critic, name)
            gradients = (
                [p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None]
                if module is not None
                else []
            )
            average_stats[f"grad/{name}"] += torch.stack(gradients).sum().sqrt() if gradients else loss.new_zeros(())
        self.optimizer.step()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.learning_rate = self.optimizer.param_groups[0]["lr"]
