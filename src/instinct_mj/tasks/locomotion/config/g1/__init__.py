"""Register Instinct Mj locomotion G1 tasks."""

from instinct_mj.tasks.registry import register_instinct_task


def _locomotion_flat_env_cfg(play: bool):
    from .flat_env_cfg import instinct_g1_locomotion_flat_env_cfg

    return instinct_g1_locomotion_flat_env_cfg(play=play)


def g1_locomotion_instinct_rl_cfg():
    from .rl_cfg import g1_locomotion_instinct_rl_cfg as _cfg

    return _cfg()


register_instinct_task(
    task_id="Instinct-Locomotion-Flat-G1-v0",
    env_cfg_factory=lambda: _locomotion_flat_env_cfg(play=False),
    play_env_cfg_factory=lambda: _locomotion_flat_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_locomotion_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-Locomotion-Flat-G1-Play-v0",
    env_cfg_factory=lambda: _locomotion_flat_env_cfg(play=True),
    play_env_cfg_factory=lambda: _locomotion_flat_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_locomotion_instinct_rl_cfg,
)


def _mild_env_cfg(play: bool):
    from .mild_env_cfg import instinct_g1_mild_env_cfg

    return instinct_g1_mild_env_cfg(play=play)


def _mild_rl_cfg():
    from .mild_rl_cfg import g1_mild_instinct_rl_cfg

    return g1_mild_instinct_rl_cfg()


register_instinct_task(
    task_id="Instinct-Locomotion-Mild-G1-v0",
    env_cfg_factory=lambda: _mild_env_cfg(play=False),
    play_env_cfg_factory=lambda: _mild_env_cfg(play=True),
    instinct_rl_cfg_factory=_mild_rl_cfg,
    env_entry_point="instinct_mj.tasks.locomotion.mild.env:MildRlEnv",
)

register_instinct_task(
    task_id="Instinct-Locomotion-Mild-G1-Play-v0",
    env_cfg_factory=lambda: _mild_env_cfg(play=True),
    play_env_cfg_factory=lambda: _mild_env_cfg(play=True),
    instinct_rl_cfg_factory=_mild_rl_cfg,
    env_entry_point="instinct_mj.tasks.locomotion.mild.env:MildRlEnv",
)


def _mild_walk_env_cfg(play: bool):
    from .mild_walk_env_cfg import instinct_g1_mild_walk_env_cfg

    return instinct_g1_mild_walk_env_cfg(play=play)


def _mild_walk_rl_cfg():
    from .mild_rl_cfg import g1_mild_walk_instinct_rl_cfg

    return g1_mild_walk_instinct_rl_cfg()


for _task_id, _play in (
    ("Instinct-Locomotion-Mild-Walk-G1-v0", False),
    ("Instinct-Locomotion-Mild-Walk-G1-Play-v0", True),
):
    register_instinct_task(
        task_id=_task_id,
        env_cfg_factory=lambda play=_play: _mild_walk_env_cfg(play=play),
        play_env_cfg_factory=lambda: _mild_walk_env_cfg(play=True),
        instinct_rl_cfg_factory=_mild_walk_rl_cfg,
        env_entry_point="instinct_mj.tasks.locomotion.mild.walking_env:MildWalkRlEnv",
    )
