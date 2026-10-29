"""Physics-rate latencies for the MILD terrain estimator task."""

import math
import re

import torch

from .walking_env import MildWalkRlEnv


class PhysicsLatency:
    """Separate per-environment motor/IMU/command delay, quantized up to dt."""

    def __init__(self, env):
        self.env = env
        self.max_lag = max(1, math.ceil(env.cfg.latency_range_ms[1] / (env.physics_dt * 1000)))
        self.size = self.max_lag + 1
        self.pointer = -1
        self.ids = torch.arange(env.num_envs, device=env.device)
        self.lags = torch.zeros((env.num_envs, 3), dtype=torch.long, device=env.device)
        self.motor_offset = torch.zeros_like(env.scene["robot"].data.joint_pos)
        self.motor_buffer = torch.zeros((self.size, *self.motor_offset.shape), device=env.device)
        self.sensor_buffer = torch.zeros((self.size, env.num_envs, 9), device=env.device)
        self.imu = torch.zeros((env.num_envs, 6), device=env.device)
        self.command = torch.zeros((env.num_envs, 3), device=env.device)
        self.pending_sensor_resets = None

    def reset(self, env_ids):
        lo, hi = self.env.cfg.latency_range_ms
        delays = torch.rand((len(env_ids), 3), device=self.env.device) * (hi - lo) + lo
        self.lags[env_ids] = torch.ceil(delays / (1000 * self.env.physics_dt)).long()
        bound = self.env.cfg.motor_offset_rad
        self.motor_offset[env_ids] = (2 * torch.rand_like(self.motor_offset[env_ids]) - 1) * bound
        data = self.env.scene["robot"].data
        self.motor_buffer[:, env_ids] = data.default_joint_pos[env_ids] + self.motor_offset[env_ids]
        self.pending_sensor_resets = env_ids

    def ensure_reset_sensors(self):
        # Observation computation follows forward() and command reset. Delay
        # backfill until then, avoiding stale pre-reset poses/commands.
        if self.pending_sensor_resets is not None:
            self.reset_sensors(self.pending_sensor_resets)
            self.pending_sensor_resets = None

    def reset_sensors(self, env_ids):
        data = self.env.scene["robot"].data
        self.imu[env_ids] = torch.cat((data.root_link_ang_vel_b, data.projected_gravity_b), dim=-1)[env_ids]
        self.command[env_ids] = self.env.command_manager.get_command("base_velocity")[env_ids]
        self.sensor_buffer[:, env_ids] = torch.cat((self.imu, self.command), dim=-1)[env_ids]

    def motor_target(self):
        """Queue desired joint positions; motor latency is command arrival delay.

        Delay the reference before feedback, rather than replaying torques
        computed from stale joint velocities (which adds feedback-loop delay).
        """
        self.pointer = (self.pointer + 1) % self.size
        data = self.env.scene["robot"].data
        self.motor_buffer[self.pointer] = data.joint_pos_target + self.motor_offset
        return self.motor_buffer[(self.pointer - self.lags[:, 0]) % self.size, self.ids]

    def sample_sensors(self):
        data = self.env.scene["robot"].data
        self.sensor_buffer[self.pointer] = torch.cat(
            (data.root_link_ang_vel_b, data.projected_gravity_b, self.env.command_manager.get_command("base_velocity")),
            dim=-1,
        )
        # Sampling is before integration: the newest sample is already dt old
        # at the next observation boundary, so lag=1 selects this sample.
        self.imu.copy_(self.sensor_buffer[(self.pointer + 1 - self.lags[:, 1]) % self.size, self.ids, :6])
        self.command.copy_(self.sensor_buffer[(self.pointer + 1 - self.lags[:, 2]) % self.size, self.ids, 6:])
        if self.env.cfg.latency_range_ms[1] == 0:
            self.imu.copy_(self.sensor_buffer[self.pointer, :, :6])
            self.command.copy_(self.sensor_buffer[self.pointer, :, 6:])


class MildVaeEnv(MildWalkRlEnv):
    def __init__(self, cfg, device, render_mode=None, **kwargs):
        super().__init__(cfg, device, render_mode, **kwargs)
        from ..config.g1.mild_vae_env_cfg import LEG_JOINTS, LEG_KD, LEG_KP

        robot = self.scene["robot"]
        self.kp = torch.zeros(robot.num_joints, device=device)
        self.kd = torch.zeros_like(self.kp)
        self.effort_limit = torch.zeros_like(self.kp)
        for expressions, kp, kd, limit in cfg.pd_gain_groups:
            for i, name in enumerate(robot.joint_names):
                if any(re.fullmatch(expr, name) for expr in expressions):
                    self.kp[i], self.kd[i], self.effort_limit[i] = kp, kd, limit
        self.leg_ids = [
            i for i, name in enumerate(robot.joint_names) if any(re.fullmatch(expr, name) for expr in LEG_JOINTS)
        ]
        if len(self.leg_ids) != 12:
            raise ValueError("Expected exactly twelve G1 leg joints")
        self.upper_ids = [i for i in range(robot.num_joints) if i not in self.leg_ids]
        # Upper-body native servos receive nominal targets; only the twelve
        # legs receive the sampled/delayed torque below.
        self.kp[self.upper_ids] = 0.0
        self.kd[self.upper_ids] = 0.0
        self.kp[self.leg_ids] = torch.tensor(LEG_KP * 2, device=device)
        self.kd[self.leg_ids] = torch.tensor(LEG_KD * 2, device=device)
        self.held_torque = torch.zeros_like(robot.data.joint_pos)
        self.control_tick = 0
        self.previous_base_velocity = robot.data.root_link_lin_vel_w.clone()
        self.latency = PhysicsLatency(self)
        self.latency.reset(torch.arange(self.num_envs, device=self.device))

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "latency"):
            self.latency.reset(env_ids)
            self.previous_base_velocity[env_ids] = 0
            self.held_torque[env_ids] = 0

    def _before_physics_step(self):
        robot = self.scene["robot"]
        data = robot.data
        data.joint_pos_target[:, self.upper_ids] = data.default_joint_pos[:, self.upper_ids]
        target = self.latency.motor_target()
        # The G1 servo evaluates fresh feedback every physics step. Setting
        # pd_decimation=10 reproduces the literal 100 Hz sampled-PD ablation;
        # its evaluation ticks are shifted by command arrival latency.
        due = (self.control_tick - self.latency.lags[:, 0]).remainder(self.cfg.pd_decimation) == 0
        torque = self.kp * (target - data.joint_pos) - self.kd * data.joint_vel
        torque = torch.clamp(torque, -self.effort_limit, self.effort_limit)
        self.held_torque.copy_(torch.where(due[:, None], torque, self.held_torque))
        robot.set_joint_effort_target(self.held_torque)
        self.control_tick = (self.control_tick + 1) % self.cfg.decimation

    def _after_physics_forward(self):
        self.latency.sample_sensors()

    def step(self, action):
        self.previous_base_velocity.copy_(self.scene["robot"].data.root_link_lin_vel_w)
        return super().step(action)
