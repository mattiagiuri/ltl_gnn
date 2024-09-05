import random
from pprint import pprint


class ReachSampler:

    @classmethod
    def partial(cls, depth: int | tuple[int, int]):
        return lambda props: cls(props, depth)

    def __init__(self, propositions: list[str], depth: int | tuple[int, int]):
        self.propositions = sorted(propositions)
        self.depth = depth

    def __call__(self) -> str:
        d = self.depth if isinstance(self.depth, int) else random.randint(*self.depth)
        props = [random.choice(self.propositions)]
        while len(props) < d:
            prop = random.choice(self.propositions)
            if props[-1] != prop:
                props.append(prop)
        formula = f'F {props[0]}'
        for p in props[1:]:
            formula = f'F ({p} & {formula})'
        return formula


if __name__ == '__main__':
    props = ['a', 'b', 'c', 'd']
    depth = (1, 3)
    sampler = ReachSampler(props, depth)

    seqs = [sampler() for _ in range(5)]
    pprint(seqs)

