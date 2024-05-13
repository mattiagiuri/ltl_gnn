import random

import numpy as np
import torch

from ltl.samplers import LTLSampler


class ReachFourSampler(LTLSampler):
    def __init__(self, propositions: list[str]):
        super().__init__(propositions)
        self.tasks = [
            f'F ({a} & F ({b} & F ({c} & F {d})))'
            for a in propositions
            for b in propositions
            for c in propositions
            for d in propositions
            if a != b and a != c and a != d and b != c and b != d and c != d
        ]

    def sample(self) -> str:
        return random.choice(self.tasks)

    def update_returns(self, task_returns: dict[str, float]):
        pass
