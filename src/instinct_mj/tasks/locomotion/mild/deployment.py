"""Restore estimator inference architecture before constructing a play environment."""

import copy
import math
from dataclasses import replace

import torch


MILD_SCHEMA_VERSION = 6
GAIT_FIELDS = (
    "gait_period",
    "gait_stance_fraction",
    "gait_clearance",
    "gait_contact_threshold",
    "randomize_gait_phase_by_env",
)
GAIT_CLOCK_CONVENTION = {
    "phase": "(episode_steps * policy_dt / gait_period + environment_offset) % 1",
    "environment_offset": "env_index * offset_per_env if randomize_gait_phase_by_env else 0",
    "offset_per_env": 0.61803398875,
    "right_foot_phase_offset": 0.5,
    "outputs": ["sin(2*pi*phase)", "cos(2*pi*phase)"],
    "moving": "norm(command_xy) > command_threshold or abs(command_yaw) > command_threshold",
    "command_threshold": 0.1,
    "standing_outputs": [0.0, 0.0],
    "reset": "episode_steps reset to zero; observation history backfilled",
}


def gait_schema(env_cfg):
    """Persist the timing and phase convention needed to reproduce policy inputs."""
    return {
        "gait": {name: getattr(env_cfg, name) for name in GAIT_FIELDS},
        "gait_clock_convention": copy.deepcopy(GAIT_CLOCK_CONVENTION),
    }


def _saved_gait_settings(schema):
    gait = schema.get("gait")
    if not isinstance(gait, dict) or set(gait) != set(GAIT_FIELDS):
        raise ValueError("Checkpoint gait configuration is missing or incomplete")
    if schema.get("gait_clock_convention") != GAIT_CLOCK_CONVENTION:
        raise ValueError("Checkpoint gait clock convention is incompatible with the current MILD task")
    for name in GAIT_FIELDS[:-1]:
        value = gait[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Checkpoint gait setting {name} must be finite")
        if value <= 0:
            raise ValueError(f"Checkpoint gait setting {name} must be positive")
    if not 0.5 < gait["gait_stance_fraction"] < 1:
        raise ValueError("Checkpoint gait_stance_fraction must be between 0.5 and 1")
    if not isinstance(gait["randomize_gait_phase_by_env"], bool):
        raise ValueError("Checkpoint randomize_gait_phase_by_env must be boolean")
    return gait


def configure_inference(env_cfg, agent_cfg, checkpoint):
    """Restore network/history/gait settings, preserving other play choices.

    This is only used by the MILD ablation runners. Gait timing is part of the
    policy's observation contract. Robot control and other observation
    conventions remain task-defined and are checked by the runner's schema.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    schema = state.get("mild_schema")
    if not isinstance(schema, dict) or schema.get("version") != MILD_SCHEMA_VERSION:
        raise ValueError(
            "MILD inference requires a version-6 gait-conditioned 29-DoF ablation checkpoint; "
            "older checkpoints without gait_clock are incompatible"
        )
    gait = _saved_gait_settings(schema)
    train_cfg = state.get("train_cfg")
    saved_policy = train_cfg.get("policy") if isinstance(train_cfg, dict) else None
    if not isinstance(saved_policy, dict) or schema.get("policy") != saved_policy:
        raise ValueError("Checkpoint policy configuration is missing or inconsistent with its MILD schema")
    if train_cfg.get("normalizers"):
        raise ValueError("MILD ablation inference does not support running observation normalization")
    policy_class = agent_cfg.policy.class_name
    if policy_class not in {"ActorCritic", "MildActorCritic"} or saved_policy.get("class_name") != policy_class:
        raise ValueError("Checkpoint policy class does not match the selected MILD ablation task")
    if env_cfg.use_vae != (policy_class == "MildActorCritic"):
        raise ValueError("Selected MILD environment and policy disagree about enabling the VAE")
    # Config serialization turns tuples into lists; restore tuple-valued
    # dataclass fields without changing the saved schema representation.
    policy_options = {
        name: (
            tuple(value)
            if isinstance(value, list) and isinstance(getattr(agent_cfg.policy, name, None), tuple)
            else value
        )
        for name, value in saved_policy.items()
    }
    try:
        policy = replace(agent_cfg.policy, **policy_options)
        restored_agent = replace(agent_cfg, policy=policy)
    except TypeError as exc:
        raise ValueError("Checkpoint policy settings are incompatible with the current MILD configuration") from exc
    if env_cfg.use_vae:
        history = saved_policy.get("history_length")
        if not isinstance(history, int) or history < 1 or history != schema.get("history_length"):
            raise ValueError("Checkpoint policy and observation history lengths are inconsistent")
        env_cfg.history_length = history
        env_cfg.observations["policy"].history_length = history
        env_cfg.observations["policy"].flatten_history_dim = True
    for name, value in gait.items():
        setattr(env_cfg, name, value)
    # Do not rerun __post_init__: doing so would also overwrite playback
    # randomization, action, command or reward choices with training defaults.
    return env_cfg, restored_agent
