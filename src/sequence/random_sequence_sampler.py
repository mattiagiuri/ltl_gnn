import random
from pprint import pprint
from typing import Callable


class RandomSequenceSampler:  # TODO: rewrite using assignments instead of simply propositions

    @classmethod
    def partial(cls, length: int, unique: bool) -> Callable[[list[str]], 'RandomSequenceSampler']:
        return lambda propositions: cls(propositions, length, unique)

    def __init__(self, propositions: list[str], length: int, unique: bool):
        self.propositions = sorted(propositions)
        self.length = length
        self.unique = unique
        self.is_adaptive = False

    def __call__(self):
        if self.unique:
            return self.sample_unique(self.length)
        return self.sample(self.length)

    def sample(self, length: int) -> list[tuple[str, str]]:
        seq = [tuple(random.sample(self.propositions, 2))]
        for i in range(1, length):
            last_reach, last_avoid = seq[-1]
            reach, avoid = random.sample(self.propositions, 2)
            while reach == last_reach or avoid == last_reach:
                reach, avoid = random.sample(self.propositions, 2)
            seq.append((reach, avoid))
        return seq

    def sample_unique(self, length: int) -> list[tuple[str, str]]:
        seq = random.sample(self.propositions, 2 * length)
        return [(seq[i], seq[i + 1]) for i in range(0, 2 * length, 2)]


if __name__ == '__main__':
    sampler = RandomSequenceSampler(['a', 'b', 'c', 'd'])
    pprint([
        sampler.sample(5)
        for _ in range(5)
    ])
