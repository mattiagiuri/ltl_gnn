import random
from pprint import pprint
from typing import Callable


class ReachSequenceSampler:  # TODO: rewrite using assignments instead of simply propositions

    @classmethod
    def partial(cls, length: int, unique: bool) -> Callable[[list[str]], 'ReachSequenceSampler']:
        return lambda propositions: cls(propositions, length, unique)

    def __init__(self, propositions: list[str], length: int, unique: bool):
        self.propositions = sorted(propositions)
        self.length = length
        self.unique = unique
        self.is_adaptive = False

    def __call__(self):
        return self.sample_unique(self.length)

    def sample_unique(self, length: int) -> list[tuple[str, str]]:
        seq = random.sample(self.propositions, length)
        return [(s, 'empty') for s in seq]


if __name__ == '__main__':
    sampler = ReachSequenceSampler(['a', 'b', 'c', 'd'], 4, True)
    pprint([
        sampler()
        for _ in range(5)
    ])
