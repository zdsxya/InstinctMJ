"""Bridge the batched MILD contact model to mjlab's current dynamics state."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
import warp as wp

from .config import MildContactCfg
from .geometry import FOOT_NAMES, sole_quadrature
from .solver import granular_law, resultant_zmp, solve_contact


class MildContact:
    def __init__(self, env, cfg: MildContactCfg):
        cfg.validate()
        self.env, self.cfg = env, cfg
        model = env.sim.mj_model
        self.body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{name}") for name in FOOT_NAMES]
        if min(self.body_ids) < 0:
            raise ValueError(f"MILD requires G1 feet {FOOT_NAMES}")
        self.body_indices = torch.tensor(self.body_ids, device=env.device)
        # Plain zero-copy torch views avoid the bridge's implicit stream switch
        # on writes during contact-graph capture. No data allocation is replaced.
        self.data = SimpleNamespace(**{
            name: wp.to_torch(getattr(env.sim.wp_data, name))
            for name in ("xmat", "xpos", "xipos", "subtree_com", "cdof", "qvel", "M", "qfrc_smooth", "xfrc_applied")
        })
        samples = [sole_quadrature(model, body, cfg.cell_size, cfg.quadrature_subdivisions) for body in self.body_ids]
        if samples[0][0].shape != samples[1][0].shape:
            raise ValueError("MILD expects matching left/right sole quadrature counts")
        self.points_local = torch.tensor(np.stack([s[0] for s in samples]), device=env.device)
        self.areas = torch.tensor(np.stack([s[1] for s in samples]), device=env.device)
        self.points_per_foot = self.points_local.shape[1]
        self.count = 2 * self.points_per_foot
        self.reference_area = cfg.reference_cell_size**2
        self.weights = self.areas.flatten()[None] / self.reference_area
        batch = env.num_envs
        kwargs = {"device": env.device, "dtype": torch.float32}
        self.positions = torch.zeros((batch, self.count, 3), **kwargs)
        self.forces = torch.zeros_like(self.positions)
        self.depth = torch.zeros((batch, self.count), **kwargs)
        self.path_length = torch.zeros_like(self.depth)
        self.previous_xy = torch.zeros((batch, self.count, 2), **kwargs)
        self.was_active = torch.zeros((batch, self.count), device=env.device, dtype=torch.bool)
        self.tangent_impulse = torch.zeros_like(self.previous_xy)
        self.previous_jacobian = torch.zeros((batch, self.count, 3, model.nv), **kwargs)
        self.has_previous = torch.zeros(batch, device=env.device, dtype=torch.bool)
        self.current_air_time = torch.zeros((batch, 2), **kwargs)
        self.current_contact_time = torch.zeros_like(self.current_air_time)
        self.force_history = torch.zeros((batch, 2, 3, 3), **kwargs)
        self.foot_force = torch.zeros((batch, 2, 3), **kwargs)
        self.foot_torque = torch.zeros_like(self.foot_force)
        self.wrench_origin = torch.zeros_like(self.foot_force)
        self.impulse_residual = torch.zeros(batch, **kwargs)
        self.normal_residual = torch.zeros(batch, **kwargs)
        self.material = {item.name: torch.zeros((batch, 1), **kwargs) for item in fields(cfg.material)}
        # Static topology, dynamic M and cdof values. Dense matrices are small
        # (35 x 35 for G1); no dense contact-space matrix is allocated.
        self.mass_rows = torch.tensor(np.repeat(np.arange(model.nv), model.M_rownnz), device=env.device)
        self.mass_cols = torch.tensor(model.M_colind.copy(), dtype=torch.long, device=env.device)
        self.mass_matrix = torch.zeros((batch, model.nv, model.nv), **kwargs)
        ancestor = np.zeros((2, model.nv), dtype=np.float32)
        for foot, body in enumerate(self.body_ids):
            ancestors = set()
            while body:
                ancestors.add(body)
                body = int(model.body_parentid[body])
            ancestor[foot] = np.isin(model.dof_bodyid, list(ancestors))
        self.ancestor = torch.tensor(ancestor, **kwargs)[None, :, None, :, None]
        self.root_ids = torch.tensor(model.body_rootid[self.body_ids].copy(), device=env.device, dtype=torch.long)
        self.graph = None
        self.reset(torch.arange(batch, device=env.device))

    def capture_graph(self):
        """Capture only tensor contact computations, not reset/randomization."""
        if not self.cfg.use_cuda_graph or not str(self.env.device).startswith("cuda"):
            return
        stream = torch.cuda.Stream(device=self.env.device)
        stream.wait_stream(torch.cuda.current_stream(self.env.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.update()
        torch.cuda.current_stream(self.env.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.update()
        torch.cuda.current_stream(self.env.device).wait_stream(stream)
        self.reset(torch.arange(self.env.num_envs, device=self.env.device))

    def step(self):
        if self.graph is None:
            self.update()
        else:
            self.graph.replay()

    def reset(self, env_ids):
        for tensor in (self.positions, self.forces, self.depth, self.path_length, self.previous_xy,
                       self.was_active, self.tangent_impulse, self.previous_jacobian, self.has_previous,
                       self.current_air_time, self.current_contact_time, self.force_history,
                       self.foot_force, self.foot_torque, self.wrench_origin, self.impulse_residual, self.normal_residual):
            tensor[env_ids] = 0
        for name, target in self.material.items():
            lo, hi = getattr(self.cfg.material, name)
            if self.cfg.randomize_material:
                target[env_ids] = torch.rand((len(env_ids), 1), device=self.env.device) * (hi - lo) + lo
            else:
                target[env_ids] = (lo + hi) * 0.5
        for body in self.body_ids:
            self.env.sim.data.xfrc_applied[env_ids, body] = 0

    def clear_applied_forces(self):
        # MILD owns only the two foot slots. Torso pushes/other bodies survive.
        self.env.sim.data.xfrc_applied[:, self.body_ids] = 0

    def kinematics(self):
        data = self.data
        batch = self.env.num_envs
        rotation = data.xmat[:, self.body_indices].reshape(batch, 2, 3, 3)
        positions = torch.einsum("bfij,fpj->bfpi", rotation, self.points_local)
        positions = positions + data.xpos[:, self.body_indices, None, :]
        offset = positions - data.subtree_com[:, self.root_ids, None, :]
        cdof = data.cdof
        angular = cdof[:, None, None, :, :3].expand(-1, 2, self.points_per_foot, -1, -1)
        linear = cdof[:, None, None, :, 3:] + torch.cross(angular, offset[:, :, :, None, :].expand_as(angular), dim=-1)
        jacobian = (linear * self.ancestor).permute(0, 1, 2, 4, 3).reshape(batch, self.count, 3, -1)
        return positions.reshape(batch, self.count, 3), jacobian

    def update(self):
        """Called once per physics substep AFTER forward and BEFORE integration."""
        dt = self.env.physics_dt
        data = self.data
        positions, jacobian = self.kinematics()
        velocity = torch.einsum("bcin,bn->bci", jacobian, data.qvel)
        bias = torch.einsum("bcin,bn->bci", (jacobian - self.previous_jacobian) / dt, data.qvel)
        bias = bias * self.has_previous[:, None, None]
        height = self.env.scene.env_origins[:, 2:3] + self.cfg.surface_height
        depth = (height - positions[:, :, 2]).clamp(min=0)
        active = depth > 0
        travelled = (positions[:, :, :2] - self.previous_xy).norm(dim=-1)
        self.path_length.copy_(torch.where(active, self.path_length + travelled * self.was_active, 0.0))
        # States exceeding this domain terminate at the policy boundary. Bound
        # the law evaluation until then, without hiding the measured depth.
        model_depth = depth.clamp(max=self.cfg.max_penetration)
        quasi, stiffness, added_mass, drag = granular_law(
            model_depth, velocity[:, :, 2], self.weights, self.material, self.reference_area
        )
        # First-order implicit evaluation of spring force at the next depth.
        force_base = (quasi - dt * stiffness * velocity[:, :, 2] + drag) * active
        effective_mass = (added_mass + dt**2 * stiffness) * active
        speed = velocity[:, :, :2].norm(dim=-1)
        resistance = self.weights * (
            self.material["drift_stiffness"] * self.path_length * model_depth
            + self.material["drift_damping"] * speed
        )
        drift = -resistance.unsqueeze(-1) * velocity[:, :, :2] / speed.clamp(min=1e-8).unsqueeze(-1)
        drift = drift * active.unsqueeze(-1)
        self.mass_matrix[:, self.mass_rows, self.mass_cols] = data.M
        self.mass_matrix[:, self.mass_cols, self.mass_rows] = data.M
        forces, impulse, residual, normal_residual = solve_contact(
            self.mass_matrix, jacobian, data.qvel, data.qfrc_smooth, bias,
            force_base, effective_mass, drift, self.material["friction"], active,
            self.tangent_impulse * self.was_active.unsqueeze(-1), dt,
            self.cfg.friction_iterations, self.cfg.coupling_iterations,
        )
        self.positions.copy_(positions)
        self.depth.copy_(depth)
        self.forces.copy_(forces)
        self.tangent_impulse.copy_(impulse)
        self.impulse_residual.copy_(residual)
        self.normal_residual.copy_(normal_residual)
        self.previous_xy.copy_(positions[:, :, :2])
        self.previous_jacobian.copy_(jacobian)
        self.was_active.copy_(active)
        self.has_previous.fill_(True)
        per_foot = forces.reshape(self.env.num_envs, 2, self.points_per_foot, 3)
        self.wrench_origin.copy_(data.xipos[:, self.body_indices])
        lever = positions.reshape_as(per_foot) - self.wrench_origin[:, :, None, :]
        self.foot_force.copy_(per_foot.sum(dim=2))
        self.foot_torque.copy_(torch.cross(lever, per_foot, dim=-1).sum(dim=2))
        data.xfrc_applied[:, self.body_indices, :3] = self.foot_force
        data.xfrc_applied[:, self.body_indices, 3:] = self.foot_torque
        contact = self.foot_force[:, :, 2] > self.cfg.contact_threshold
        self.current_contact_time.copy_(torch.where(contact, self.current_contact_time + dt, 0.0))
        self.current_air_time.copy_(torch.where(contact, 0.0, self.current_air_time + dt))
        self.force_history[:, :, 1:] = self.force_history[:, :, :-1].clone()
        self.force_history[:, :, 0] = self.foot_force

    def metrics(self):
        com = self.wrench_origin
        zmp, support = resultant_zmp(
            self.foot_force, self.foot_torque, com,
            self.env.scene.env_origins[:, 2:3] + self.cfg.surface_height,
            max(self.cfg.contact_threshold, 1e-6),
        )
        return {
            "Mild/max_depth": self.depth.amax().detach(),
            "Mild/mean_depth": self.depth.mean().detach(),
            "Mild/max_foot_force": self.foot_force.norm(dim=-1).amax().detach(),
            "Mild/mean_path": self.path_length.mean().detach(),
            "Mild/impulse_residual": self.impulse_residual.mean().detach(),
            "Mild/normal_residual_N": self.normal_residual.mean().detach(),
            "Mild/not_converged": (self.impulse_residual > self.cfg.impulse_tolerance).float().mean().detach(),
            "Mild/max_zmp_offset": ((zmp[..., :2] - com[..., :2]).norm(dim=-1) * support).amax().detach(),
        }

    def debug_vis(self, visualizer):
        if not self.cfg.debug_vis:
            return
        for env_id in visualizer.get_env_indices(self.env.num_envs):
            points = self.positions[env_id].detach().cpu().numpy()
            forces = self.forces[env_id].detach().cpu().numpy()
            for point, force in zip(points, forces):
                if np.linalg.norm(force) > 0.1:
                    visualizer.add_arrow(point, point + force * 0.002, (0.2, 0.9, 0.3, 0.8), width=0.002)
