import random

import numpy as np
import torch

from ltl.samplers import LTLSampler


class EventuallySampler(LTLSampler):
    def __init__(self, propositions: list[str]):
        super().__init__(propositions)
        self.temperature = 0.5
        self.returns = None

    def sample(self) -> str:
        if self.returns is None:
            return f'F {random.choice(self.propositions)}'
        assert len(self.returns) == len(self.propositions)
        probs = self.compute_sampling_prob()
        return f'F {np.random.choice(self.propositions, p=probs)}'

    def compute_sampling_prob(self) -> np.ndarray:
        rets = sorted(self.returns.items(), key=lambda kv: kv[0])
        rets = torch.tensor([r[1] for r in rets])
        assert (rets <= 1).all().item()
        probs = torch.nn.functional.softmax(-rets / self.temperature, dim=0)
        return probs.numpy()

    def update_returns(self, returns: dict[str, float]):
        if self.returns is None:
            self.returns = returns
        else:
            self.returns.update(returns)
