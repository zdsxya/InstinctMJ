"""Flat reward semantics using MILD's actual support forces."""

import torch


def feet_air_time_positive_biped(env, command_name: str, threshold: float):
    contact = env.mild_contact
    in_contact = contact.current_contact_time > 0
    mode_time = torch.where(in_contact, contact.current_contact_time, contact.current_air_time)
    single_stance = in_contact.sum(dim=1) == 1
    reward = torch.where(single_stance[:, None], mode_time, 0.0).amin(dim=1).clamp(max=threshold)
    return reward * (env.command_manager.get_command(command_name)[:, :2].norm(dim=-1) > 0.1)


def feet_slide(env, asset_cfg, threshold: float = 0.1):
    contact = env.mild_contact.force_history.norm(dim=-1).amax(dim=-1) > threshold
    velocity = env.scene[asset_cfg.name].data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
    return (velocity.norm(dim=-1) * contact).sum(dim=1)


def excessive_penetration(env):
    return env.mild_contact.depth.amax(dim=-1) > env.cfg.deformable_contact.max_penetration


def fallen(env):
    # A deeply yielding surface must not hide a fallen robot from native
    # body-contact termination. projected_gravity_b.z = -1 when upright.
    return env.scene["robot"].data.projected_gravity_b[:, 2] > -0.3
