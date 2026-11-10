"""VAE observation adapter using the standard MJLab history/reset lifecycle."""

from instinct_mj.rl.vecenv_wrapper import InstinctRlVecEnvWrapper


class MildVaeVecEnvWrapper(InstinctRlVecEnvWrapper):
    """Store actor history and same-time critic labels via regular PPO storage.

    History is already provided by ObservationManager. The critic contains its
    original observations plus state_targets for supervised losses; there is
    no need to duplicate actor history in the critic stream.
    """
