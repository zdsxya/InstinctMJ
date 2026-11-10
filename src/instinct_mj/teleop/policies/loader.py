"""Load task actors without constructing training algorithms or AMP datasets.

Instinct-RL PPO/AMP and Parkour SuperMimic share the ``model_state_dict``
checkpoint format. Their actor architecture and observation normalizers come
from the task registration; optimizers, discriminators and world models are
not needed for inference.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from instinct_rl import modules

from instinct_mj.tasks.registry import load_instinct_rl_cfg


class InferencePolicy:
    """Deterministic actor with frozen observation normalization and reset."""

    def __init__(
        self,
        actor_critic: torch.nn.Module,
        normalizer: torch.nn.Module | None,
        *,
        num_observations: int,
        num_actions: int,
        device: str | torch.device,
    ):
        self.actor_critic = actor_critic.to(device).eval()
        self.normalizer = normalizer.to(device).eval() if normalizer is not None else None
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.device = torch.device(device)

    @torch.inference_mode()
    def __call__(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.num_observations:
            raise ValueError(
                f"Policy expects observations [batch, {self.num_observations}], "
                f"received {tuple(observations.shape)}. Check the task observation configuration."
            )
        observations = observations.to(self.device)
        if self.normalizer is not None:
            observations = self.normalizer(observations)
        actions = self.actor_critic.act_inference(observations)
        expected_shape = (observations.shape[0], self.num_actions)
        if tuple(actions.shape) != expected_shape:
            raise ValueError(f"Policy actions have shape {tuple(actions.shape)}; expected {expected_shape}.")
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Policy returned nonfinite actions; simulation was not advanced.")
        return actions

    @torch.inference_mode()
    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Clear recurrent actor state whenever the simulation is reset."""
        reset = getattr(self.actor_critic, "reset", None)
        if reset is not None:
            reset(dones.to(self.device) if dones is not None else None)


class ZeroPolicy:
    """Zero joint-action input for testing simulation wiring without a model."""

    def __init__(self, num_actions: int, device: str | torch.device):
        self.num_actions = num_actions
        self.device = torch.device(device)

    def __call__(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.zeros((observations.shape[0], self.num_actions), device=self.device)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones


def _policy_observation_size(obs_format: Mapping[str, Any]) -> int:
    segments = obs_format.get("policy")
    if not isinstance(segments, Mapping) or not segments:
        raise ValueError("The task must expose a nonempty 'policy' observation group.")
    for name, shape in segments.items():
        if not shape or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise ValueError(f"Invalid policy observation shape for {name!r}: {shape!r}")
    return sum(math.prod(shape) for shape in segments.values())


def _load_normalizer(
    config: Mapping[str, Any], checkpoint: Mapping[str, Any], num_observations: int, device: str | torch.device
) -> torch.nn.Module | None:
    policy_config = (config.get("normalizers") or {}).get("policy")
    state_key = "policy_normalizer_state_dict"
    if policy_config is None:
        if state_key in checkpoint:
            raise ValueError(
                "Checkpoint contains a policy observation normalizer, but the selected task does not configure one. "
                "Use the task and model configuration that produced this checkpoint."
            )
        return None
    policy_config = copy.deepcopy(policy_config)
    normalizer_class = policy_config.pop("class_name")
    if normalizer_class is None:
        if state_key in checkpoint:
            raise ValueError("Checkpoint contains normalization statistics but policy normalization is disabled.")
        return None
    if state_key not in checkpoint:
        raise ValueError(
            f"Checkpoint is missing {state_key!r}, required by the selected task. "
            "Refusing to run with uninitialized normalization statistics."
        )
    normalizer = modules.build_normalizer(
        input_shape=num_observations,
        normalizer_class_name=normalizer_class,
        normalizer_kwargs=policy_config,
    ).to(device)
    try:
        normalizer.load_state_dict(checkpoint[state_key], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Policy normalization statistics do not match the task observations: {exc}") from exc
    return normalizer.eval()


def load_policy(
    task_id: str,
    checkpoint_file: str | Path,
    vec_env: Any,
    device: str | torch.device,
) -> InferencePolicy:
    """Load an InstinctMJ ``.pt`` actor for an already constructed environment.

    Weight and normalizer dimensions are checked strictly. Legacy checkpoints
    do not store semantic observation/joint ordering, so callers must select the
    original task and retain its observation/action definitions.
    """
    path = Path(checkpoint_file).expanduser().resolve()
    if path.suffix != ".pt":
        raise ValueError("Teleoperation currently loads InstinctMJ .pt checkpoints; ONNX is not supported yet.")
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # Training checkpoints also contain optimizers/discriminators. Keep those
    # unused tensors off the GPU; only the actor and policy normalizer move.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("model_state_dict"), Mapping):
        raise ValueError(
            f"{path} is not an InstinctMJ policy checkpoint: expected 'model_state_dict'. "
            "Select model_<iteration>.pt, not the SuperMimic training-state sidecar."
        )

    config = copy.deepcopy(load_instinct_rl_cfg(task_id).to_dict())
    if config.get("ckpt_manipulator"):
        raise ValueError("Teleoperation requires a compatible checkpoint without a training checkpoint manipulator.")
    obs_format = vec_env.get_obs_format()
    num_observations = _policy_observation_size(obs_format)
    policy_config = config["policy"]
    actor_critic = modules.build_actor_critic(
        policy_config.pop("class_name"),
        policy_config,
        obs_format,
        num_actions=vec_env.num_actions,
        num_rewards=vec_env.num_rewards,
    ).to(device)
    try:
        actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(
            f"Checkpoint {path.name} does not match task {task_id!r}. "
            f"Check actor configuration, observation history and joint/action dimensions.\n{exc}"
        ) from exc
    normalizer = _load_normalizer(config, checkpoint, num_observations, device)
    policy = InferencePolicy(
        actor_critic,
        normalizer,
        num_observations=num_observations,
        num_actions=vec_env.num_actions,
        device=device,
    )
    policy.reset()
    print(f"[INFO] Loaded teleop policy: {path}")
    return policy
