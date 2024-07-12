import numpy as np

import preprocessing
from model.model import Model
from sequence.search import SequenceSearch


class Agent:
    def __init__(self, model: Model, search: SequenceSearch, verbose=False):
        self.model = model
        self.sequence = None
        self.search = search
        self.verbose = verbose

    def reset(self):
        self.sequence = None

    def get_action(self, obs, info, deterministic=False) -> np.ndarray:
        if 'ldba_state_changed' in info:
            self.sequence = self.search(obs['ldba'], obs['ldba_state'], obs)
            if self.verbose:
                print(f'Selected sequence: {self.sequence}')
        assert self.sequence is not None
        obs['goal'] = self.sequence
        return self.forward(obs, deterministic)

    def forward(self, obs, deterministic=False) -> np.ndarray:
        if not (isinstance(obs, list) or isinstance(obs, tuple)):
            obs = [obs]
        preprocessed = preprocessing.preprocess_obss(obs)
        dist, value = self.model(preprocessed)
        action = dist.mode if deterministic else dist.sample()
        return action.detach().numpy()
