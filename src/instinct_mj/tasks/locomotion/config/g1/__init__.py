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


def _mild_vae_env_cfg(play: bool, variant: str):
    from .mild_vae_env_cfg import instinct_g1_mild_vae_env_cfg

    return instinct_g1_mild_vae_env_cfg(play=play, variant=variant)


def _mild_vae_rl_cfg(variant: str):
    from .mild_vae_rl_cfg import g1_mild_vae_instinct_rl_cfg

    return g1_mild_vae_instinct_rl_cfg(variant)


# The same 29-DoF locomotion task and budget across the estimator ablations.
# Existing Mild-Walk keeps its separate AMP policy and registration above.
for _variant, _suffix in (("baseline", "-Baseline"), ("vae", ""), ("full", "-Full")):
    for _play in (False, True):
        register_instinct_task(
            task_id=f"Instinct-Locomotion-Mild-VAE{_suffix}-G1{'-Play' if _play else ''}-v0",
            env_cfg_factory=lambda play=_play, variant=_variant: _mild_vae_env_cfg(play, variant),
            play_env_cfg_factory=lambda variant=_variant: _mild_vae_env_cfg(True, variant),
            instinct_rl_cfg_factory=lambda variant=_variant: _mild_vae_rl_cfg(variant),
            env_entry_point="instinct_mj.tasks.locomotion.mild.vae_env:MildVaeEnv",
            vecenv_entry_point=(
                "instinct_mj.rl:InstinctRlVecEnvWrapper"
                if _variant == "baseline"
                else "instinct_mj.tasks.locomotion.mild.vae_wrapper:MildVaeVecEnvWrapper"
            ),
            runner_entry_point=(
                "instinct_mj.tasks.locomotion.mild.runner:MildBaselineRunner"
                if _variant == "baseline"
                else "instinct_mj.tasks.locomotion.mild.runner:MildVaeRunner"
            ),
        )
