"""MILD Table I rewards and explicitly separated estimator supervision."""

import math

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from .walking_mdp import desired_support, foot_phase, moving

# Same order as MildMaterialCfg; scale SI quantities without running statistics.
MATERIAL_SCALES = (1.0, 1.0, 1.0, 0.1, 1.0, 0.001, 1.0, 0.001, 1e-5, 0.001, 1.0)


def state_targets(env):
    """[base velocity, left/right sole velocity, left/right sole height].

    Velocities are world-relative, expressed in the root-link frame. Heights
    are signed relative to the UNDEFORMED surface, in metres.
    Sample current kinematics rather than last substep's contact diagnostics.
    """
    if not hasattr(env, "mild_contact"):
        return torch.zeros((env.num_envs, 11), device=env.device)
    contact = env.mild_contact
    positions, jacobian = contact.kinematics()
    velocities = torch.einsum("bcin,bn->bci", jacobian, contact.data.qvel)
    weights = contact.areas / contact.areas.sum(-1, keepdim=True)
    velocity_w = (velocities.reshape(env.num_envs, 2, -1, 3) * weights[None, :, :, None]).sum(2)
    quaternion = env.scene["robot"].data.root_link_quat_w[:, None].expand(-1, 2, -1)
    velocity_b = quat_apply_inverse(quaternion.reshape(-1, 4), velocity_w.reshape(-1, 3))
    height = (positions[..., 2].reshape(env.num_envs, 2, -1) * weights).sum(-1)
    height = height - env.scene.env_origins[:, 2:3] - contact.cfg.surface_height
    return torch.cat((env.scene["robot"].data.root_link_lin_vel_b, velocity_b.reshape(env.num_envs, 6), height), dim=-1)


def terrain_parameters(env):
    if not hasattr(env, "mild_contact"):
        return torch.zeros((env.num_envs, 22), device=env.device)
    material = torch.cat(
        [value * scale for value, scale in zip(env.mild_contact.material.values(), MATERIAL_SCALES, strict=True)],
        dim=-1,
    )
    return material.repeat(1, 2)  # Main experiment: same material beneath both feet.


def delayed_angular_velocity(env):
    if hasattr(env, "latency") and env.cfg.latency_range_ms[1] > 0:
        env.latency.ensure_reset_sensors()
        return env.latency.imu[:, :3]
    return env.scene["robot"].data.root_link_ang_vel_b


def delayed_gravity(env):
    if hasattr(env, "latency") and env.cfg.latency_range_ms[1] > 0:
        env.latency.ensure_reset_sensors()
        return env.latency.imu[:, 3:]
    return env.scene["robot"].data.projected_gravity_b


def delayed_command(env):
    if hasattr(env, "latency") and env.cfg.latency_range_ms[1] > 0:
        env.latency.ensure_reset_sensors()
        return env.latency.command
    return env.command_manager.get_command("base_velocity")


def gait_alignment(env):
    contact = env.mild_contact.foot_force[..., 2] > env.cfg.gait_contact_threshold
    return 1.3 * (contact == desired_support(env)).float().sum(-1) - 0.3


def clearance_error(env):
    phase = foot_phase(env)
    swing = ~desired_support(env)
    progress = ((phase - env.cfg.gait_stance_fraction) / (1 - env.cfg.gait_stance_fraction)).clamp(0, 1)
    target = env.cfg.gait_clearance * torch.sin(math.pi * progress)
    height = state_targets(env)[:, 9:]
    # Correct the printed positive-error reward: minimizing error is intended.
    return ((height - target).abs() * swing).sum(-1) * moving(env)


def base_acceleration_reward(env):
    return torch.exp(-3 * (env.scene["robot"].data.root_link_lin_vel_w - env.previous_base_velocity).norm(dim=-1))


def action_smoothness(env):
    current, previous = env.action_manager.action, env.action_manager.prev_action
    return (
        (current - previous).square() + (current - 2 * previous + env.action_manager.prev_prev_action).square()
    ).sum(-1)


def foot_slip(env):
    velocity = state_targets(env)[:, 3:9].reshape(-1, 2, 3)
    contact = env.mild_contact.foot_force[..., 2] > env.cfg.gait_contact_threshold
    return (velocity.norm(dim=-1) * contact).sum(-1)
