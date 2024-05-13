import random

import numpy as np
import torch

from ltl.samplers import LTLSampler


class LoopSampler(LTLSampler):
    def __init__(self, propositions: list[str]):
        super().__init__(propositions)
        self.tasks = [
            f'GF {a} & GF {b}'
            for a in propositions
            for b in propositions
            if a != b
        ]

    def sample(self) -> str:
        return random.choice(self.tasks)
