"""Edit MILD-Walk teleoperation defaults here; CLI flags override these values."""

from instinct_mj.teleop.config import CameraConfig, ControlConfig, SceneConfig, TeleopConfig


def default_config() -> TeleopConfig:
    return TeleopConfig(
        task="mild_walk",
        control=ControlConfig(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(0.0, 0.0),  # Current walking policy was not trained for side stepping.
            ang_vel_z=(-1.0, 1.0),
            linear_acceleration=1.0,
            yaw_acceleration=2.0,
        ),
        scene=SceneConfig(terrain="plane"),
        camera=CameraConfig(distance=3.0, elevation=-15.0, azimuth=135.0),
    )
