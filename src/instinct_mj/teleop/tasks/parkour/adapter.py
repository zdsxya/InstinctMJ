"""Preserve depth/history observations while replacing target-driven commands."""

from instinct_mj.teleop.config import TeleopConfig
from instinct_mj.teleop.tasks.base import TaskAdapter, configure_reset_events


class ParkourAdapter(TaskAdapter):
    def configure_task(self, env_cfg, cfg: TeleopConfig) -> None:
        configure_reset_events(env_cfg, cfg, joint_position_range=(0.0, 0.0))
        terrain = env_cfg.scene.terrain
        generator = terrain.terrain_generator
        if cfg.scene.terrain != "task":
            if cfg.scene.terrain not in generator.sub_terrains:
                raise ValueError(
                    f"Unknown parkour terrain {cfg.scene.terrain!r}; choose task or "
                    + ", ".join(generator.sub_terrains)
                )
            generator.sub_terrains = {cfg.scene.terrain: generator.sub_terrains[cfg.scene.terrain]}
            generator.sub_terrains[cfg.scene.terrain].proportion = 1.0
        generator.curriculum = False
        generator.num_rows = cfg.scene.terrain_rows
        generator.num_cols = cfg.scene.terrain_cols
        generator.difficulty_range = (cfg.scene.terrain_difficulty, cfg.scene.terrain_difficulty)
        generator.seed = cfg.seed
        # Target patch sampling and virtual leg-obstacle scoring are train-only.
        # Terrain geometry, generated spawn origins and camera preprocessing stay intact.
        for sub_terrain in generator.sub_terrains.values():
            sub_terrain.flat_patch_sampling = None
        terrain.max_init_terrain_level = 0
        terrain.collision_debug_vis = cfg.scene.debug_vis
        terrain.virtual_obstacles = {}
        env_cfg.scene.sensors = tuple(
            sensor for sensor in env_cfg.scene.sensors
            if sensor.name in ("camera", "torso_contact_forces")
        )
