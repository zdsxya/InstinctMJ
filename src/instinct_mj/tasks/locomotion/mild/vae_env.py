"""Estimator diagnostics with the existing full-body G1 actuator pipeline."""

import torch

from .walking_env import MildWalkRlEnv


class MildVaeEnv(MildWalkRlEnv):
    """Reuse MILD gait diagnostics and native 29-DoF joint-position control.

    There is no estimator-specific actuator override. History is owned by
    ObservationManager and shares the regular environment reset lifecycle.
    """

    def step(self, action):
        # Measure the rollout state before stepping, so auto-reset cannot
        # substitute a new episode's state or command for a terminal sample.
        command = self.command_manager.get_command("base_velocity")
        velocity = self.scene["robot"].data.root_link_lin_vel_b
        tracking_error = (velocity[:, :2] - command[:, :2]).norm(dim=-1).mean().detach()
        result = super().step(action)
        terminated, timed_out = result[2:4]
        fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for name in ("fallen", "base_contact"):
            if name in self.termination_manager.active_terms:
                fallen |= self.termination_manager.get_term(name)
        self.extras["step"].update({
            "Evaluation/Velocity_Tracking_Error": tracking_error,
            "Evaluation/Fallen_Episodes": fallen.sum().float().detach(),
            "Evaluation/Completed_Episodes": (terminated | timed_out).sum().float().detach(),
        })
        return result
