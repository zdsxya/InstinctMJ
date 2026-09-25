"""Add walking diagnostics without changing the contact integration loop."""

from .env import MildRlEnv
from .walking_mdp import walking_metrics


class MildWalkRlEnv(MildRlEnv):
    def step(self, action):
        result = super().step(action)
        self.extras["step"].update(walking_metrics(self))
        return result
