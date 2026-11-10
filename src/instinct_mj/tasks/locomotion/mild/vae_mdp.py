"""Training-only state labels for the full-body G1 proprioceptive estimator."""

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse


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
