import random
from typing import Callable

import numpy as np
import torch

from ltl.samplers import LTLSampler


class FixedSampler(LTLSampler):
    @classmethod
    def partial_from_formula(cls, formula: str) -> Callable[[list[str]], LTLSampler]:
        return lambda props: cls(formula, props)

    def __init__(self, formula: str, propositions: list[str]):
        super().__init__(propositions)
        self.formula = formula

    def sample(self) -> str:
        return self.formula
