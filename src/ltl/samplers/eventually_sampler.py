import random

import numpy as np
import torch

from ltl.samplers import LTLSampler


class EventuallySampler(LTLSampler):
    def __init__(self, propositions: list[str]):
        super().__init__(propositions)
        self.beta = 6.0
        self.returns = None

    def sample(self) -> str:
        if self.returns is None:
            return f'F {random.choice(self.propositions)}'
        probs = self.compute_sampling_prob()
        return f'F {np.random.choice(self.propositions, p=probs)}'

    def compute_sampling_prob(self) -> np.ndarray:
        returns = sorted(self.returns.items(), key=lambda kv: kv[0])
        returns = torch.tensor([r[1] for r in returns])
        assert (returns <= 1).all().item()
        probs = torch.nn.functional.softmax(self.beta * (1 - returns), dim=0)
        return probs.numpy()
