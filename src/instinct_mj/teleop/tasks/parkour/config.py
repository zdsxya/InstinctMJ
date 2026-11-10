"""Edit Parkour teleoperation defaults here; PPO and SuperMimic share them."""

from instinct_mj.teleop.config import CameraConfig, ControlConfig, SceneConfig, TeleopConfig


def default_config() -> TeleopConfig:
    return TeleopConfig(
        task="parkour_amp",
        control=ControlConfig(
            lin_vel_x=(0.0, 0.8),  # Current parkour commands train forward motion only.
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            linear_acceleration=0.8,
            yaw_acceleration=2.0,
        ),
        scene=SceneConfig(
            terrain="perlin_rough",
            terrain_rows=1,
            terrain_cols=1,
            terrain_difficulty=0.2,
        ),
        camera=CameraConfig(distance=4.0, elevation=-20.0, azimuth=135.0),
    )
