import random
from pprint import pprint
from typing import Callable

from ltl.automata import LDBASequence
from ltl.logic import Assignment, FrozenAssignment
from envs.chessworld import ChessWorld8

chessworld = ChessWorld8()
props = set(chessworld.get_propositions())
assignments = chessworld.get_possible_assignments()
assignments.remove(Assignment.zero_propositions(chessworld.get_propositions()))
all_assignments = [frozenset([a.to_frozen()]) for a in assignments]
pieces_assignments = []


def get_complete_piece_assignments(color: str) -> frozenset[FrozenAssignment]:
    color_assignments = []
    for assignment in chessworld.get_possible_assignments():
        if color in assignment.get_true_propositions():
            color_assignments.append(assignment.to_frozen())
    return frozenset(color_assignments)


for piece in ['queen', 'rook', 'bishop', 'knight', 'pawn']:
    complete_piece = get_complete_piece_assignments(piece)
    # print(piece)
    # print(complete_piece)
    all_assignments.append(complete_piece)
    pieces_assignments.append(complete_piece)


def chessworld8_all_reach_tasks(depth: int) -> Callable:
    def wrapper(propositions: list[str]) -> list[LDBASequence]:
        reachs = [(a, frozenset()) for a in all_assignments]

        def rec(depth: int):
            if depth == 0:
                return []
            if depth == 1:
                return [[r] for r in reachs]
            rec_res = rec(depth - 1)
            result = []
            for task in rec_res:
                next_reach = task[0][0]
                for p, _ in reachs:
                    if next_reach.issubset(p):
                        continue
                    result.append([(p, frozenset())] + task)
            return result

        return [LDBASequence(task) for task in rec(depth)]

    return wrapper


def chessworld8_sample_reach(depth: int | tuple[int, int]) -> Callable:
    def wrapper(propositions: list[str]) -> LDBASequence:
        d = random.randint(*depth) if isinstance(depth, tuple) else depth
        reach = random.choice(all_assignments)
        task = [(reach, frozenset())]
        for _ in range(d - 1):
            reach = random.choice([a for a in all_assignments if not reach.issubset(a)])
            task.append((reach, frozenset()))
        return LDBASequence(task)

    return wrapper


def chessworld8_all_reach_avoid():
    def wrapper(_):
        seqs = []
        for reach in all_assignments:
            available = [a for a in all_assignments if a != reach and not reach.issubset(a)]
            for avoid in available:
                seqs.append(LDBASequence([(reach, avoid)]))
        return seqs
    return wrapper


def chessworld8_sample_reach_avoid(
        depth: int | tuple[int, int],
        num_reach: int | tuple[int, int],
        num_avoid: int | tuple[int, int],
        not_reach_same_as_last: bool = False
) -> Callable[[list[str]], LDBASequence]:
    def wrapper(propositions: list[str]) -> LDBASequence:
        def sample_one(last_reach):
            nr = random.randint(*num_reach) if isinstance(num_reach, tuple) else num_reach
            na = random.randint(*num_avoid) if isinstance(num_avoid, tuple) else num_avoid
            available = [a for a in all_assignments if a not in last_reach] if not_reach_same_as_last else all_assignments
            reach = random.sample(available, nr)
            available = [a for a in all_assignments if a not in reach and a not in last_reach]
            available = [a for a in available if
                         not any([r.issubset(a) for r in reach])
                         and (len(last_reach) == 0 or not any(
                             [r.issubset(a) for r in last_reach]))]
            if len(available) < na:
                if isinstance(num_avoid, tuple):
                    na = random.randint(num_avoid[0], len(available)) if num_avoid[0] < len(available) else len(
                        available)
                else:
                    raise ValueError('Not enough propositions to sample from')
            avoid = random.sample(available, na)
            reach_assignments = frozenset.union(*reach)
            avoid_assignments = frozenset.union(*avoid) if len(avoid) > 0 else frozenset()
            return reach_assignments, avoid_assignments, reach

        d = random.randint(*depth) if isinstance(depth, tuple) else depth
        last_reach = []
        seq = []
        for _ in range(d):
            reach, avoid, reach_props = sample_one(last_reach)
            seq.append((reach, avoid))
            last_reach = reach_props
        return LDBASequence(seq)

    return wrapper


def chessworld8_sample_reach_stay(num_stay: int, num_avoid: tuple[int, int]) -> Callable[[list[str]], LDBASequence]:
    def wrapper(propositions: list[str]) -> LDBASequence:
        nr = random.choice([1, 2])
        reach = random.sample(pieces_assignments, nr)
        reach_assignments = frozenset.union(*reach)

        na = random.randint(*num_avoid)
        available = [a for a in all_assignments if a not in reach and all(not r.issubset(a) for r in reach)]
        avoid = random.sample(available, na)
        avoid = frozenset.union(*avoid) if len(avoid) > 0 else frozenset()

        second_avoid = frozenset.union(*all_assignments).difference(reach_assignments).union({Assignment.zero_propositions(propositions).to_frozen()})

        task = [(LDBASequence.EPSILON, avoid), (reach_assignments, second_avoid)]
        return LDBASequence(task, repeat_last=num_stay)

    return wrapper


def count_reach_avoid():
    import itertools
    import math

    tot = 0

    for nr_1 in [1, 2]:
        first_reaches = itertools.combinations(all_assignments, nr_1)

        if nr_1 == 2:
            first_reaches = [(i, j) for i, j in first_reaches if not (i.issubset(j) or j.issubset(i))]

        for last_reach in first_reaches:
            available_avoid_first = [a for a in all_assignments if a not in last_reach]
            available_avoid_first = [a for a in available_avoid_first if
                                     not any([r.issubset(a) for r in last_reach])
                                     ]

            for na_1 in [0, 1, 2]:

                c1 = math.comb(len(available_avoid_first), na_1)

                # Count depth = 1
                tot += c1

                for nr_2 in [1, 2]:
                    second_reaches = itertools.combinations(all_assignments, nr_2)

                    if nr_2 == 2:
                        second_reaches = [(i, j) for i, j in second_reaches if not (i.issubset(j) or j.issubset(i))]

                    for reach in second_reaches:
                        available = [a for a in all_assignments if a not in reach and a not in last_reach]
                        available = [a for a in available if
                                     not any([r.issubset(a) for r in reach])
                                     and (len(last_reach) == 0 or not any(
                                         [r.issubset(a) for r in last_reach]))]

                        for na_2 in [0, 1, 2]:
                            c2 = math.comb(len(available), na_2)
                            tot += (c1 * c2)

    return tot


if __name__ == '__main__':
    print(all_assignments)
    print(len(all_assignments))
    print(len(chessworld8_all_reach_tasks(1)(["a"])))
    print(len(chessworld8_all_reach_avoid()([])))

    print(count_reach_avoid())
