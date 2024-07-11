import numpy as np

import preprocessing
from model.model import Model


class SequenceAgent:
    def __init__(self, model: Model, verbose=False):
        self.model = model
        self.sequence = None
        self.index = 0
        self.verbose = verbose

    def reset(self):
        self.sequence = None
        self.index = 0

    def get_action(self, obs, info, deterministic=False) -> np.ndarray:
        if 'ldba_state_changed' in info:
            if self.sequence is None:
                self.sequence = self.get_sequence(obs['goal'])
            else:
                props = info['propositions']
                reach, avoid = self.sequence[self.index]
                if reach in props:
                    self.index += 1
                else:
                    if avoid in props:
                        self.index -= 1
        assert self.sequence is not None
        obs['goal'] = self.sequence[self.index:]
        return self.forward(obs, deterministic)

    def get_sequence(self, f):
        seq = []
        collect = []
        for c in f:
            if 'a' <= c <= 'z':
                collect.append(c)
            if len(collect) == 2:
                seq.append((collect[1], collect[0]))
                collect = []
        return seq

    def forward(self, obs, deterministic=False) -> np.ndarray:
        if not (isinstance(obs, list) or isinstance(obs, tuple)):
            obs = [obs]
        preprocessed = preprocessing.preprocess_obss(obs)
        dist, value = self.model(preprocessed)
        action = dist.mode if deterministic else dist.sample()
        return action.detach().numpy()
