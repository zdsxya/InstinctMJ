"""Store the identical causal actor history alongside privileged critic data."""

import torch

from instinct_mj.rl.vecenv_wrapper import InstinctRlVecEnvWrapper


class MildVaeVecEnvWrapper(InstinctRlVecEnvWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_critic_obs += self.num_obs

    def get_obs_format(self):
        result = super().get_obs_format()
        result["critic"]["policy_history"] = (self.num_obs,)
        return result

    def get_obs_segments(self, group_name="policy"):
        return self.get_obs_format()[group_name]

    def _pack_observations(self, obs_dict):
        packed = super()._pack_observations(obs_dict)
        packed["critic"] = torch.cat((packed["critic"], packed["policy"]), dim=-1)
        return packed
