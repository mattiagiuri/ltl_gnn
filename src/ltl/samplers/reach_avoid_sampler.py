import random
from typing import Callable

from ltl.automata import ltl2ldba
from ltl.samplers import LTLSampler


class ReachAvoidSampler(LTLSampler):

    @classmethod
    def partial_from_depth(cls, depth: int) -> Callable[[list[str]], LTLSampler]:
        return lambda props: cls(props, depth)

    def __init__(self, propositions: list[str], depth: int = 1):
        super().__init__(propositions)
        self.tasks = [
            ReachAvoidSampler.create_task(combination)
            for combination in ReachAvoidSampler.get_possible_combinations(propositions, depth)
        ]

    @staticmethod
    def create_task(combination: list[tuple[str, str]]) -> str:
        if len(combination) == 1:
            avoid, reach = combination[0]
            return f'!{avoid} U {reach}'
        avoid, reach = combination[0]
        return f'!{avoid} U ({reach} & ({ReachAvoidSampler.create_task(combination[1:])}))'

    @staticmethod
    def get_possible_combinations(propositions: list[str], depth: int) -> list[list[tuple[str, str]]]:
        if depth == 1:
            return [[(a, b)] for a in propositions for b in propositions if a != b]
        rest = ReachAvoidSampler.get_possible_combinations(propositions, depth - 1)
        result = []
        for seq in rest:
            next_avoid, next_reach = seq[0]
            for a in propositions:
                for b in propositions:
                    if a != b and a != next_avoid and a != next_reach and b != next_avoid and b != next_reach:
                        result.append([(a, b)] + seq)
        return result

    def sample(self) -> str:
        return random.choice(self.tasks)


if __name__ == '__main__':
    sampler = ReachAvoidSampler(['a', 'b', 'c', 'd'], depth=2)
    for task in sampler.tasks:
        ltl2ldba(task)