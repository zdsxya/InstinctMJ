"""MILD Table I / Fig. 4 locomotion, with twelve leg actions and no AMP."""

import math
from dataclasses import dataclass, field

from mjlab.actuator import BuiltinMotorActuatorCfg, BuiltinPdActuatorCfg
from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg

from instinct_mj.tasks.locomotion.mild import vae_mdp

from .mild_env_cfg import G1MildEnvCfg, configure_mild_play

LEG_JOINTS = (".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint")
# The paper gives six gains but no joint ordering. This G1 adaptation uses
# hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll on each leg.
LEG_KP = (50.0, 50.0, 70.0, 70.0, 20.0, 20.0)
LEG_KD = (5.0, 5.0, 7.0, 7.0, 0.2, 0.2)


@dataclass(kw_only=True)
class G1MildVaeEnvCfg(G1MildEnvCfg):
    pd_gain_groups: tuple = field(default_factory=tuple)
    pd_decimation: int = 1  # G1 inner servo at 1 kHz; use 10 for literal 100 Hz ablation.
    history_length: int = 10
    latency_range_ms: tuple[float, float] = (1.0, 10.0)
    motor_offset_rad: float = 0.035
    gait_period: float = 0.8
    gait_stance_fraction: float = 0.5
    gait_clearance: float = 0.065
    gait_contact_threshold: float = 5.0
    randomize_gait_phase_by_env: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.history_length < 2 or not 0 <= self.latency_range_ms[0] <= self.latency_range_ms[1]:
            raise ValueError("Invalid estimator history/latency configuration")
        self.run_name = "G1MildVAEChange1"
        self.scene.num_envs = 4096
        self.decimation = 10  # 100 Hz position commands, integrated at 1 kHz.
        if self.pd_decimation < 1 or self.decimation % self.pd_decimation:
            raise ValueError("PD decimation must be a positive divisor of the policy decimation")
        self.actions["joint_pos"].actuator_names = LEG_JOINTS
        self.actions["joint_pos"].scale = 1.0  # Actions are radian deviations, as Sec. II-C1.
        robot = self.scene.entities["robot"]
        robot.init_state.joint_pos.update(
            {
                ".*_elbow_joint": 1.4,
                "left_shoulder_pitch_joint": 0.0,
                "right_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.3,
                "right_shoulder_roll_joint": -0.3,
            }
        )
        # Cache the G1 upper-body gains before replacing implicit PD with motors.
        # Guard against tyro calling __post_init__ again on an existing config.
        if not self.pd_gain_groups:
            self.pd_gain_groups = tuple(
                (a.target_names_expr, a.stiffness, a.damping, a.effort_limit) for a in robot.articulation.actuators
            )
        actuator_configs = []
        for old in robot.articulation.actuators:
            expressions = old.target_names_expr
            common = dict(
                target_names_expr=expressions,
                effort_limit=old.effort_limit,
                armature=old.armature,
                frictionloss=old.frictionloss,
                viscous_damping=old.viscous_damping,
            )
            if any(any(part in expr for part in ("hip", "knee", "ankle")) for expr in expressions):
                actuator_configs.append(BuiltinMotorActuatorCfg(**common))
            else:
                # Extra G1 joints are outside the twelve-action SA01 policy.
                # Native servo holds them at rest without introducing a delayed,
                # unactuated 100 Hz oscillation into the torso/arm dynamics.
                kp, kd = (
                    (100.0, 6.0)
                    if any("waist" in expr for expr in expressions)
                    else (20.0, 0.5) if all("wrist" in expr for expr in expressions) else (50.0, 2.0)
                )
                actuator_configs.append(BuiltinPdActuatorCfg(stiffness=kp, damping=kd, **common))
        robot.articulation.actuators = tuple(actuator_configs)
        self.observations["policy"].history_length = self.history_length
        policy = self.observations["policy"].terms
        policy["base_ang_vel"].func = vae_mdp.delayed_angular_velocity
        policy["projected_gravity"].func = vae_mdp.delayed_gravity
        policy["velocity_commands"] = ObservationTermCfg(func=vae_mdp.delayed_command)
        for group in ("policy", "critic"):
            terms = self.observations[group].terms
            terms["base_ang_vel"].scale = 0.25
            terms["joint_vel"].scale = 0.05
            for name in ("joint_pos", "joint_vel"):
                terms[name].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
        critic = self.observations["critic"].terms
        critic.pop("base_lin_vel", None)
        critic["state_targets"] = ObservationTermCfg(func=vae_mdp.state_targets)
        critic["terrain_parameters"] = ObservationTermCfg(func=vae_mdp.terrain_parameters)
        # Explicit allowlist: no inherited posture, AMP, or contact-shaping terms.
        self.rewards = {name: self.rewards[name] for name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp")}
        self.rewards["track_lin_vel_xy_exp"].weight = 1.4
        self.rewards["track_ang_vel_z_exp"].weight = 1.1
        for term in self.rewards.values():
            term.params["std"] = math.sqrt(1 / 5)
        self.rewards.update(
            {
                "gait_alignment": RewardTermCfg(func=vae_mdp.gait_alignment, weight=1.4),
                "feet_clearance": RewardTermCfg(func=vae_mdp.clearance_error, weight=-2.4),
                "feet_slide": RewardTermCfg(func=vae_mdp.foot_slip, weight=-0.1),
                "base_acceleration": RewardTermCfg(func=vae_mdp.base_acceleration_reward, weight=0.2),
                "action_rate_l2": RewardTermCfg(func=vae_mdp.action_smoothness, weight=-0.003),
            }
        )
        command = self.commands["base_velocity"]
        command.heading_command = False
        command.ranges.heading = None
        command.rel_standing_envs = 0.1
        command.ranges.lin_vel_x = (0.0, 1.2)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-1.5, 1.5)
        self.events["physics_material"] = None  # MILD samples the foot material.
        self.events["push_robot"] = None
        self.events["base_external_force_torque"] = None
        self.events["add_base_mass"].params["mass_distribution_params"] = (-4.0, 4.0)
        self.events["base_com"] = EventTermCfg(
            func=dr.body_com_offset,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "ranges": (-0.006, 0.006),
                "operation": "add",
            },
        )
        # Nonnegative subset: negative cg/cd are incompatible with this solver's
        # passive positive-depth convention. This remains an explicit deviation.
        material = self.deformable_contact.material
        material.alpha = (0.698, 1.310)
        material.recruitment = (0.8, 2.4)
        material.mass_scale = (0.0, 4.6)
        material.drag_scale = (0.0, 28.0)
        material.friction = (0.2, 1.3)
        material.penetration_stiffness = (8.0, 1400.0)
        material.conical_stress = (1e4, 2.4e5)


def instinct_g1_mild_vae_env_cfg(*, play=False):
    cfg = G1MildVaeEnvCfg()
    if play:
        configure_mild_play(cfg)
        cfg.events["base_com"] = None
        cfg.latency_range_ms = (0.0, 0.0)
        cfg.motor_offset_rad = 0.0
    return cfg
