import random
from pprint import pprint


class FixedSequenceSampler:  # TODO: rewrite using assignments instead of simply propositions
    def __init__(self, propositions: list[str], seq: list[tuple[str, str]]):
        self.seq = seq

    def sample(self) -> list[tuple[str, str]]:
        return self.seq
