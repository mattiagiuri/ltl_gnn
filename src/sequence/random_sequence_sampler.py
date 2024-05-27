import random
from pprint import pprint


class RandomSequenceSampler:  # TODO: rewrite using assignments instead of simply propositions
    def __init__(self, propositions: list[str]):
        self.propositions = sorted(propositions)
        self.is_adaptive = False

    def sample(self, length: int) -> list[tuple[str, str]]:
        seq = [tuple(random.sample(self.propositions, 2))]
        seq[0] = (seq[0][0], 'empty')
        for i in range(1, length):
            last_reach, last_avoid = seq[-1]
            reach, avoid = random.sample(self.propositions, 2)
            while reach == last_reach or avoid == last_reach:
                reach, avoid = random.sample(self.propositions, 2)
            seq.append((reach, 'empty'))
        return seq


if __name__ == '__main__':
    sampler = RandomSequenceSampler(['a', 'b', 'c', 'd'])
    pprint([
        sampler.sample(5)
        for _ in range(5)
    ])
