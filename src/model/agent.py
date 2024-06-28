import copy

import numpy as np
import torch

import preprocessing
from model.model import Model


class Agent:
    def __init__(self, model: Model):
        self.model = model
        self.sequence = None

    def get_action(self, obs, deterministic=False, shielding=False) -> np.ndarray:
        if not 'changed' in obs:
            self.sequence = obs['goal']
        else:
            if obs['changed']:
                seq_to_value = {}
                for seq in obs['sequences']:
                    obs['goal'] = seq
                    seq_to_value[tuple(seq)] = self.get_value(obs, deterministic, shielding)
                self.sequence = max(seq_to_value, key=seq_to_value.get)
        assert self.sequence is not None
        obs['goal'] = self.sequence
        return self.get_action_value(obs, deterministic, shielding)[0]

    def get_value(self, obs, deterministic=False, shielding=False) -> float:
        return self.get_action_value(obs, deterministic, shielding)[1]

    def get_action_value(self, obs, deterministic=False, shielding=False) -> tuple[np.ndarray, float]:
        if not (isinstance(obs, list) or isinstance(obs, tuple)):
            obs = [obs]
        preprocessed = preprocessing.preprocess_obss(obs)
        dist, value = self.model(preprocessed)
        if shielding:
            action = self.get_shielded_action(obs, dist, deterministic)
        else:
            action = dist.mode if deterministic else dist.sample()
        return action.detach().numpy(), value.item()

    def get_shielded_action(self, obs, dist, deterministic, max_tries=100, threshold=1):
        assert len(obs) == 1
        obs = obs[0]
        goal, avoid = obs['goal'][0]
        assert avoid != 'empty'
        modified_obs = copy.deepcopy(obs)
        modified_obs['goal'] = [(goal, 'empty')]
        preprocessed = preprocessing.preprocess_obss([modified_obs])

        action = dist.mode if deterministic else dist.sample()
        actions = sorted(range(4), key=lambda x: dist.probs[0, x].item(), reverse=True)
        actions = [torch.tensor(a).unsqueeze(0) for a in actions]
        assert actions[0] == dist.mode
        for a in actions:
            q_value = self.model.forward_q(preprocessed, a)
            if q_value.item() < threshold:
                return a
        # tries = 0
        # while tries < max_tries:
        #     q_value = self.model.forward_q(preprocessed, action)
        #     if q_value.item() < threshold:
        #         # if tries > 10:
        #         #    print(f"Shielding succeeded after {tries} tries.")
        #         return action
        #     action = dist.sample()
        #     tries += 1
        # print(f"Shielding failed.")
        return dist.mode if deterministic else dist.sample()
