"""Clock-conditioned walking rewards; all support comes from the MILD solver."""

import math

import torch


def moving(env):
    command = env.command_manager.get_command("base_velocity")
    return command[:, :2].norm(dim=-1) > 0.1


def foot_phase(env):
    # Spread training environments across the cycle without mutable clock state.
    # Episode resets also reset the clock; environment zero starts in double support.
    offset = torch.arange(env.num_envs, device=env.device) * 0.61803398875
    phase = env.episode_length_buf * env.step_dt / env.cfg.gait_period + offset
    return torch.stack((phase, phase + 0.5), dim=-1).remainder(1.0)


def gait_clock(env):
    phase = foot_phase(env)[:, 0] * (2 * math.pi)
    return torch.stack((phase.sin(), phase.cos()), dim=-1) * moving(env)[:, None]


def desired_support(env):
    return (foot_phase(env) < env.cfg.gait_stance_fraction) | ~moving(env)[:, None]


def phase_contact(env):
    force = env.mild_contact.foot_force[:, :, 2].clamp(min=0)
    stance = desired_support(env)
    # Smooth force scores avoid rewarding tiny residual forces as full support.
    support_score = 1.0 - torch.exp(-force / 40.0)
    swing_score = torch.exp(-force / 10.0)
    return torch.where(stance, support_score, swing_score).mean(dim=-1)


def swing_clearance(env):
    phase = foot_phase(env)
    swing = (phase >= env.cfg.gait_stance_fraction) & moving(env)[:, None]
    swing_phase = ((phase - env.cfg.gait_stance_fraction) / (1 - env.cfg.gait_stance_fraction)).clamp(0, 1)
    target = env.cfg.gait_clearance * torch.sin(math.pi * swing_phase)
    contact = env.mild_contact
    # Use the lowest sole cell, not the ankle origin; stance depth is unconstrained.
    height = contact.positions[:, :, 2].reshape(env.num_envs, 2, -1).amin(dim=-1)
    height = height - env.scene.env_origins[:, 2:3] - contact.cfg.surface_height
    score = torch.exp(-((height - target) / 0.025).square())
    return (score * swing).sum(dim=-1) / swing.sum(dim=-1).clamp(min=1)


def flight(env):
    support = env.mild_contact.foot_force[:, :, 2] > env.cfg.gait_contact_threshold
    return (~support.any(dim=-1)).float()


def walking_metrics(env):
    # Exclude reset states and initial settling from gait-quality statistics.
    valid = (~env.reset_buf.bool()) & (env.episode_length_buf * env.step_dt > 1.0) & moving(env)
    count = valid.sum()
    denominator = count.clamp(min=1)

    def mean(value):
        return (value * valid).sum().detach() / denominator

    support = env.mild_contact.foot_force[:, :, 2] > env.cfg.gait_contact_threshold
    velocity = env.scene["robot"].data.root_link_lin_vel_b
    command = env.command_manager.get_command("base_velocity")
    return {
        "Walk/sample_count": count.detach(),
        "Walk/flight_fraction": mean((support.sum(-1) == 0).float()),
        "Walk/single_support_fraction": mean((support.sum(-1) == 1).float()),
        "Walk/double_support_fraction": mean((support.sum(-1) == 2).float()),
        "Walk/phase_agreement": mean((support == desired_support(env)).float().mean(-1)),
        "Walk/vx_error_abs": mean((velocity[:, 0] - command[:, 0]).abs()),
        "Walk/vz_squared": mean(env.scene["robot"].data.root_link_lin_vel_w[:, 2].square()),
    }
