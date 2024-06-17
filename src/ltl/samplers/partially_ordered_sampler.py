import random


class PartiallyOrderedSampler:

    def __init__(self, propositions: list[str], depth: int, num_conjuncts: int, disjunct_prob=0.25):
        self.propositions = sorted(propositions)
        self.depth = depth
        self.num_conjuncts = num_conjuncts
        self.disjunct_prob = disjunct_prob

    def __call__(self) -> str:
        formula = f'({self.sample_conjunct()})'
        for _ in range(self.num_conjuncts - 1):
            formula += f' & ({self.sample_conjunct()})'
        return formula

    def sample_conjunct(self) -> str:
        seq = self.sample_sequence()
        seq.reverse()
        prop_to_str = lambda prop: prop[0] if len(prop) == 1 else f'({prop[0]} | {prop[1]})'
        conjunct = f'F {prop_to_str(seq[0])}'
        for prop in seq[1:]:
            conjunct = f'F ({prop_to_str(prop)} & {conjunct})'
        return conjunct

    def sample_sequence(self) -> list[list[str]]:
        seq = []
        for _ in range(self.depth):
            population = [p for p in self.propositions if len(seq) == 0 or p not in seq[-1]]
            num_sample = 2 if random.random() < self.disjunct_prob else 1
            seq.append(random.sample(population, num_sample))
        return seq


if __name__ == '__main__':
    sampler = PartiallyOrderedSampler(['a', 'b', 'c', 'd'], 3, 2)
    for _ in range(5):
        print(sampler())
