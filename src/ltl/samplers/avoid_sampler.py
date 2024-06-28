import random


class AvoidSampler:

    @classmethod
    def partial(cls, depth: int, num_conjuncts: int):
        return lambda props: cls(props, depth, num_conjuncts)

    def __init__(self, propositions: list[str], depth: int, num_conjuncts: int):
        self.propositions = sorted(propositions)
        self.depth = depth
        self.num_conjuncts = num_conjuncts
        if 2 * self.depth * self.num_conjuncts > len(self.propositions):
            raise ValueError('Not enough propositions to sample from')

    def __call__(self) -> str:
        num_props = 2 * self.depth * self.num_conjuncts
        props = random.sample(self.propositions, num_props)
        formula = ''
        for i in range(self.num_conjuncts):
            conjunct = f'!{props.pop()} U {props.pop()}'
            for _ in range(self.depth - 1):
                conjunct = f'!{props.pop()} U ({props.pop()} & ({conjunct}))'
            formula += f'({conjunct})'
            if i < self.num_conjuncts - 1:
                formula += ' & '
        return formula


if __name__ == '__main__':
    props = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
    depth = 6
    num_conjuncts = 1
    sampler = AvoidSampler(props, depth, num_conjuncts)

    seqs = [sampler() for _ in range(2)]
    print(seqs)

