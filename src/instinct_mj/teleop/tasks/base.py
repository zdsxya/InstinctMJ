"""Small shared preparation helpers; physical integration stays with each task."""

from __future__ import annotations

from mjlab.viewer.viewer_config import ViewerConfig

from instinct_mj.teleop.commands.velocity import ManualVelocityCommandCfg
from instinct_mj.teleop.config import TeleopConfig


class TaskAdapter:
    def __init__(self, task_id: str):
        self.task_id = task_id

    def prepare(self, cfg: TeleopConfig):
        import instinct_mj.tasks  # noqa: F401 — register original environment factories
        from instinct_mj.tasks.registry import load_env_cfg

        env_cfg = load_env_cfg(self.task_id, play=False)
        env_cfg.scene.num_envs = 1
        env_cfg.seed = cfg.seed
        env_cfg.auto_reset = False
        # Keep the native observation/action contract, including task history,
        # scales, actuator delays and policy step duration. AMP is train-only.
        env_cfg.observations = {
            name: group for name, group in env_cfg.observations.items() if name in ("policy", "critic")
        }
        for group in env_cfg.observations.values():
            group.enable_corruption = False
            for term in group.terms.values():
                if "debug_vis" in term.params:
                    term.params["debug_vis"] = False
        env_cfg.rewards = {}
        env_cfg.curriculum = {}
        env_cfg.monitors = {}
        env_cfg.scene.sensors = tuple(
            sensor for sensor in env_cfg.scene.sensors if sensor.name != "motion_reference"
        )
        for sensor in env_cfg.scene.sensors:
            if hasattr(sensor, "debug_vis"):
                sensor.debug_vis = cfg.scene.debug_vis
        # Timeout and exhausted expert clips do not end an interactive session.
        # Physical failures and terrain bounds remain task-owned terminations.
        env_cfg.terminations = {
            name: term for name, term in env_cfg.terminations.items()
            if term is not None and name not in ("time_out", "dataset_exhausted")
        }
        env_cfg.commands = {"base_velocity": ManualVelocityCommandCfg(control=cfg.control)}
        env_cfg.viewer.origin_type = (
            ViewerConfig.OriginType.ASSET_ROOT if cfg.camera.follow else ViewerConfig.OriginType.WORLD
        )
        env_cfg.viewer.entity_name = "robot" if cfg.camera.follow else None
        env_cfg.viewer.body_name = None
        for name in ("distance", "elevation", "azimuth", "lookat"):
            setattr(env_cfg.viewer, name, getattr(cfg.camera, name))
        self.configure_task(env_cfg, cfg)
        return env_cfg

    def configure_task(self, env_cfg, cfg: TeleopConfig) -> None:
        raise NotImplementedError

    @staticmethod
    def bind_command(env):
        """Return the actual manager-owned command; never overwrite an obs slice."""
        return env.unwrapped.command_manager.get_term("base_velocity")


def configure_reset_events(env_cfg, cfg: TeleopConfig, *, joint_position_range: tuple[float, float]) -> None:
    """Retain deterministic reset events only, removing training randomization."""
    env_cfg.events = {
        key: env_cfg.events[key] for key in ("reset_base", "reset_robot_joints")
    }
    x, y, z = cfg.scene.spawn_offset
    env_cfg.events["reset_base"].params["pose_range"] = {
        "x": (x, x), "y": (y, y), "z": (z, z),
        "yaw": (cfg.scene.spawn_yaw, cfg.scene.spawn_yaw),
    }
    env_cfg.events["reset_base"].params["velocity_range"] = {}
    env_cfg.events["reset_robot_joints"].params.update(
        position_range=joint_position_range, velocity_range=(0.0, 0.0)
    )
