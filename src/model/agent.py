import numpy as np
import torch

import preprocessing
from model.model import Model


class Agent:
    def __init__(self, model: Model):
        self.model = model

    def get_action(self, obs: dict[str, np.ndarray | str], deterministic=False) -> np.ndarray:
        preprocessed = preprocessing.preprocess_obss([obs])
        dist, _ = self.model(preprocessed)
        action = dist.mean if deterministic else dist.sample()
        return action.flatten().detach().numpy()
