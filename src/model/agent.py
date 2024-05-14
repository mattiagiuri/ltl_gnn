import numpy as np

import preprocessing
from model.model import Model


class Agent:
    def __init__(self, model: Model):
        self.model = model

    def get_action(self, obs, deterministic=False) -> np.ndarray:
        if not (isinstance(obs, list) or isinstance(obs, tuple)):
            obs = [obs]
        preprocessed = preprocessing.preprocess_obss(obs)
        dist, _ = self.model(preprocessed)
        action = dist.mode if deterministic else dist.sample()
        return action.detach().numpy()
