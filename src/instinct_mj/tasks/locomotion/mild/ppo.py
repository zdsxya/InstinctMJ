"""MILD Eq. 16 joint optimization, without an imitation discriminator."""

import torch
from instinct_rl.algorithms.ppo import PPO


class MildVaePPO(PPO):
    def __init__(self, *args, regression_weight=0.3, vae_kl_weight=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < regression_weight < 1 or vae_kl_weight < 0:
            raise ValueError("Invalid MILD joint loss weights")
        self.regression_weight, self.vae_kl_weight = regression_weight, vae_kl_weight
        # PPO.update multiplies these exactly once.
        self.surrogate_loss_coef = 1 - regression_weight
        self.value_loss_coef *= 1 - regression_weight
        self.entropy_coef *= 1 - regression_weight
        self.regression_loss_coef = regression_weight

    def compute_losses(self, minibatch):
        losses, intermediate, stats = super().compute_losses(minibatch)
        auxiliary = self.actor_critic.auxiliary_losses(minibatch.obs, minibatch.critic_obs)
        losses["regression_loss"] = sum(
            value * (self.vae_kl_weight if name == "vae_kl" else 1.0) for name, value in auxiliary.items()
        )
        stats.update({f"estimator/{name}": value.detach() for name, value in auxiliary.items()})
        stats["ppo_ratio_deviation"] = (intermediate["ratio"].detach() - 1).abs().mean()
        if "kl" in intermediate:
            stats["ppo_kl"] = intermediate["kl"].detach().mean()
        return losses, intermediate, stats

    def gradient_step(self, loss, average_stats):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite MILD joint loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm, error_if_nonfinite=True
        )
        average_stats["grad_norm"] += norm.detach()
        for name in ("encoder", "state_head", "latent_mean", "decoder", "modulation", "actor", "critic"):
            gradients = [
                p.grad.detach().square().sum()
                for p in getattr(self.actor_critic, name).parameters()
                if p.grad is not None
            ]
            average_stats[f"grad/{name}"] += torch.stack(gradients).sum().sqrt()
        self.optimizer.step()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.learning_rate = self.optimizer.param_groups[0]["lr"]
