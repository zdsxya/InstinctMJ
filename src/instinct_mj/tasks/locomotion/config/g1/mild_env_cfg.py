"""MILD contact ablation: flat observations, actions, rewards and ordinary PPO."""

from dataclasses import dataclass, field

from mjlab.managers import TerminationTermCfg

from instinct_mj.tasks.locomotion.mild import mdp
from instinct_mj.tasks.locomotion.mild.config import MildContactCfg
from instinct_mj.tasks.locomotion.mild.geometry import mild_g1_spec

from .flat_env_cfg import G1LocomotionFlatEnvCfg


@dataclass(kw_only=True)
class G1MildEnvCfg(G1LocomotionFlatEnvCfg):
    deformable_contact: MildContactCfg = field(default_factory=MildContactCfg)

    def __post_init__(self):
        super().__post_init__()
        self.scene.entities["robot"].spec_fn = mild_g1_spec
        self.scene.sensors = tuple(s for s in self.scene.sensors if s.name != "feet_contact_forces")
        self.sim.mujoco.timestep = 0.001
        self.decimation = 20  # Preserve flat's 50 Hz policy.
        self.run_name = "G1MildContactPPO"
        air = self.rewards["feet_air_time"]
        air.func = mdp.feet_air_time_positive_biped
        air.params.pop("sensor_name", None)
        slide = self.rewards["feet_slide"]
        slide.func = mdp.feet_slide
        slide.params.pop("sensor_cfg", None)
        self.terminations["excessive_penetration"] = TerminationTermCfg(func=mdp.excessive_penetration)
        self.terminations["fallen"] = TerminationTermCfg(func=mdp.fallen)


def instinct_g1_mild_env_cfg(*, play: bool = False) -> G1MildEnvCfg:
    cfg = G1MildEnvCfg()
    if play:
        configure_mild_play(cfg)
    return cfg


def configure_mild_play(cfg: G1MildEnvCfg) -> None:
    """Shared deterministic evaluation setup for both MILD experiments."""
    cfg.scene.num_envs = 1
    cfg.deformable_contact.randomize_material = False
    cfg.deformable_contact.debug_vis = True
    cfg.observations["policy"].enable_corruption = False
    # Explicitly disable randomization, unlike flat-play's partial override.
    cfg.events["physics_material"] = None
    cfg.events["add_base_mass"] = None
    cfg.events["push_robot"] = None
    cfg.events["base_external_force_torque"] = None
    cfg.events["reset_base"].params["pose_range"] = {}
    cfg.events["reset_base"].params["velocity_range"] = {}
    cfg.events["reset_robot_joints"].params.update(position_range=(1.0, 1.0), velocity_range=(0.0, 0.0))
    command = cfg.commands["base_velocity"]
    command.heading_command = False
    command.ranges.heading = None
    command.rel_standing_envs = 0.0
    command.ranges.lin_vel_x = (0.6, 0.6)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
