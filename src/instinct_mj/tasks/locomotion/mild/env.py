"""Task-local substep integration, preserving mjlab's manager/reset contract."""

import torch

from instinct_mj.envs.manager_based_rl_env import InstinctRlEnv

from .contact import MildContact


class MildRlEnv(InstinctRlEnv):
    def __init__(self, cfg, device, render_mode=None, **kwargs):
        super().__init__(cfg, device, render_mode, **kwargs)
        self.mild_contact = MildContact(self, cfg.deformable_contact)
        self.sim.forward()
        self.mild_contact.capture_graph()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "mild_contact"):
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            self.mild_contact.reset(ids)

    def step(self, action):
        # Kept in sync with mjlab 1.6 ManagerBasedRlEnv.step. A control-step
        # event cannot replace this hook: contact is updated at every substep.
        if not self.cfg.auto_reset and torch.any(self._manual_reset_pending):
            raise RuntimeError("Reset terminated MILD environments before stepping again")
        self.extras["log"] = {}
        # The runner retains each step dictionary until the PPO update. A new
        # container prevents later steps overwriting all earlier diagnostics.
        self.extras["step"] = {}
        self.action_manager.process_action(action.to(self.device))
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self._before_physics_step()
            self.scene.write_data_to_sim()
            self.mild_contact.clear_applied_forces()
            # Refresh cdof, M, actuation, and bias with CURRENT qpos/qvel and
            # without the previous contact force. Native body contacts remain.
            self.sim.forward()
            self._after_physics_forward()
            self.mild_contact.step()
            self.sim.step()
            self.scene.update(dt=self.physics_dt)
            self.metrics_manager.compute_substep()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        # Derived body poses/velocities must match the final integrated state
        # for rewards and supervised state targets, not lag by one substep.
        self.sim.forward()
        if not torch.isfinite(self.sim.data.qpos).all() or not torch.isfinite(self.mild_contact.forces).all():
            raise FloatingPointError("Nonfinite MILD dynamics; inspect material ranges and solver diagnostics")
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self.metrics_manager.compute()
        contact_metrics = self.mild_contact.metrics()
        if "step" in self.event_manager.available_modes:
            self.event_manager.apply(mode="step", dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if self.cfg.auto_reset and len(reset_ids):
            self.recorder_manager.record_pre_reset(reset_ids)
            self._reset_idx(reset_ids)
            self.scene.write_data_to_sim()
        self.sim.forward()
        if self.cfg.auto_reset and len(reset_ids):
            self._command_dt.fill_(self.step_dt)
            self._command_dt[reset_ids] = 0.0
            self.command_manager.compute(dt=self._command_dt)
        else:
            self.command_manager.compute(dt=self.step_dt)
        self.sim.sense()
        self.obs_buf = self.observation_manager.compute(update_history=True)
        if self.cfg.auto_reset and len(reset_ids):
            self.recorder_manager.record_post_reset(reset_ids)
        elif len(reset_ids):
            self._manual_reset_pending[reset_ids] = True
        self.recorder_manager.record_post_step()
        self.extras.setdefault("step", {}).update(contact_metrics)
        self.extras["step"].update(self.monitor_manager.update(dt=self.step_dt))
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def update_visualizers(self, visualizer):
        super().update_visualizers(visualizer)
        if hasattr(self, "mild_contact"):
            self.mild_contact.debug_vis(visualizer)

    def _before_physics_step(self):
        """Task-local actuator processing at the physics rate."""

    def _after_physics_forward(self):
        """Task-local sensor sampling at the physics rate."""
