import random
from typing import Callable

from ltl.automata import LDBASequence
from ltl.logic import Assignment


def sample_reach_avoid(
        depth: int | tuple[int, int],
        num_reach: int | tuple[int, int],
        num_avoid: int | tuple[int, int],
) -> Callable[[list[str]], LDBASequence]:
    def wrapper(propositions: list[str]) -> LDBASequence:
        def sample_one(last_reach: set[str]):
            nr = random.randint(*num_reach) if isinstance(num_reach, tuple) else num_reach
            na = random.randint(*num_avoid) if isinstance(num_avoid, tuple) else num_avoid
            available = [p for p in propositions if p not in last_reach]
            reach = random.sample(available, nr)
            available = [p for p in available if p not in reach]
            avoid = random.sample(available, na)
            assert not (set(reach) & set(avoid) or set(reach) & set(last_reach) or set(avoid) & set(last_reach))
            reach_assignments = frozenset([Assignment.single_proposition(p, propositions).to_frozen() for p in reach])
            avoid_assignments = frozenset([Assignment.single_proposition(p, propositions).to_frozen() for p in avoid])
            return reach_assignments, avoid_assignments, reach

        d = random.randint(*depth) if isinstance(depth, tuple) else depth
        last_reach = set()
        seq = []
        for _ in range(d):
            reach, avoid, reach_props = sample_one(last_reach)
            seq.append((reach, avoid))
            last_reach = reach_props
        return tuple(seq)

    return wrapper


def all_reach_avoid_tasks(depth: int) -> Callable[[list[str]], list[LDBASequence]]:
    def wrapper(propositions: list[str]) -> list[LDBASequence]:
        reach_avoids = [(frozenset([Assignment.single_proposition(p, propositions).to_frozen()]),
                         frozenset([Assignment.single_proposition(q, propositions).to_frozen()]))
                        for p in propositions for q in propositions if p != q]

        def rec(depth: int) -> list[LDBASequence]:
            if depth == 0:
                return []
            if depth == 1:
                return [(ra,) for ra in reach_avoids]
            rec_res = rec(depth - 1)
            result = []
            for task in rec_res:
                next_reach = task[0][0]
                for p, q in reach_avoids:
                    if p == next_reach or q == next_reach:
                        continue
                    result.append(((p, q),) + task)
            return result

        return rec(depth)

    return wrapper


if __name__ == '__main__':
    tasks = all_reach_avoid_tasks(1)(['a', 'b', 'c'])
    print(tasks)
