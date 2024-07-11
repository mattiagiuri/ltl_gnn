import random
from pprint import pprint


class AvoidSampler:

    @classmethod
    def partial(cls, depth: int, num_conjuncts: int):
        return lambda props: cls(props, depth, num_conjuncts)

    def __init__(self, propositions: list[str], depth: int, num_conjuncts: int):
        self.propositions = sorted(propositions)
        self.depth = depth
        self.num_conjuncts = num_conjuncts
        # if 2 * self.depth * self.num_conjuncts > len(self.propositions):
        #     raise ValueError('Not enough propositions to sample from')

    def __call__(self) -> str:
        d = self.depth if isinstance(self.depth, int) else random.randint(*self.depth)
        n = self.num_conjuncts if isinstance(self.num_conjuncts, int) else random.randint(*self.num_conjuncts)
        num_props = 2 * d * n
        props = random.sample(self.propositions, num_props)
        formula = ''
        for i in range(n):
            conjunct = f'!{props.pop()} U {props.pop()}'
            for _ in range(d - 1):
                conjunct = f'!{props.pop()} U ({props.pop()} & ({conjunct}))'
            formula += f'({conjunct})'
            if i < n - 1:
                formula += ' & '
        return formula


if __name__ == '__main__':
    props = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
    depth = (1, 3)
    num_conjuncts = (1, 2)
    sampler = AvoidSampler(props, depth, num_conjuncts)

    seqs = [sampler() for _ in range(5)]
    pprint(seqs)

