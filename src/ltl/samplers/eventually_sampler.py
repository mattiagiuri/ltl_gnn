import random

from ltl.samplers import LTLSampler


class EventuallySampler(LTLSampler):
    def __init__(self, propositions: list[str]):
        super().__init__(propositions)

    def sample(self) -> str:
        return f'F {random.choice(self.propositions)}'
