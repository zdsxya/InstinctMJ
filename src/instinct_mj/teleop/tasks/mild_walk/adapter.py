"""MILD uses its original environment and contact solver, with manual commands."""

from instinct_mj.teleop.config import TeleopConfig
from instinct_mj.teleop.tasks.base import TaskAdapter, configure_reset_events


class MildWalkAdapter(TaskAdapter):
    def configure_task(self, env_cfg, cfg: TeleopConfig) -> None:
        if cfg.scene.terrain not in ("task", "plane"):
            raise ValueError("MILD-Walk teleop supports scene.terrain=plane (or task) only")
        configure_reset_events(env_cfg, cfg, joint_position_range=(1.0, 1.0))
        env_cfg.deformable_contact.randomize_material = False
        env_cfg.deformable_contact.debug_vis = cfg.scene.debug_vis
