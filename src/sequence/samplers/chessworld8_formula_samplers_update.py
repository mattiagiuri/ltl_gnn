import itertools
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
complete_piece_assignments = []

complete_assignment = frozenset.union(*all_assignments)

sample_voc = {0: 'PAD', 1: 'EPSILON', 2: 'NULL', 3: 'queen', 4: 'rook', 5: 'knight', 6: 'bishop', 7: 'pawn',
                    8: 'queen&rook', 9: 'queen&bishop', 10: 'queen&pawn&bishop', 11: 'queen&pawn&rook',
                    12: 'knight&rook', 13: 'bishop&rook', 14: 'knight&bishop', 15: 'blank'}


def get_complete_piece_assignments(color: str) -> frozenset[FrozenAssignment]:
    color_assignments = []
    for assignment in chessworld.get_possible_assignments():
        if color in assignment.get_true_propositions():
            color_assignments.append(assignment.to_frozen())
    return frozenset(color_assignments)


for piece in ['rook', 'knight', 'pawn', 'queen', 'bishop']:
    cur_piece_assignment = get_complete_piece_assignments(piece)
    all_assignments.append(cur_piece_assignment)
    complete_piece_assignments.append(cur_piece_assignment)


reach_ors = {i: [frozenset.union(*x) for x in itertools.combinations(complete_piece_assignments, i)] for i in range(1, 3)}
all_ors = reach_ors[1] + reach_ors[2]

all_ands = []

for i in range(2, 3):
    for tup in itertools.combinations(complete_piece_assignments, i):
        and_set = frozenset.intersection(*tup)

        if len(and_set) > 0 and and_set not in all_ands:
            all_ands.append(and_set)

# print(len(all_ands))
# print(all_ands)

all_pairs = itertools.combinations(complete_piece_assignments, 2)

reach_x_and_not_y = [x - y for x, y in all_pairs if len(x & y) > 0] + [y - x for x, y in all_pairs if len(x & y) > 0]

all_reach = all_ands + reach_x_and_not_y + all_ors

all_reach_difficult = all_ors + all_ands

# print(len(all_reach))


def chessworld_all_reach_tasks(depth: int) -> Callable:
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


def chessworld_sample_reach_update(depth: int | tuple[int, int]) -> Callable:
    def wrapper(propositions: list[str]) -> LDBASequence:
        d = random.randint(*depth) if isinstance(depth, tuple) else depth
        reach = random.choice(all_reach)
        task = [(reach, frozenset())]
        for _ in range(d - 1):
            reach = random.choice([a for a in all_reach if not reach.issubset(a)])
            task.append((reach, frozenset()))
        return LDBASequence(task)

    return wrapper


def chessworld_all_reach_avoid():
    def wrapper(_):
        seqs = []
        for reach in all_assignments:
            available = [a for a in all_assignments if a != reach and not reach.issubset(a)]
            for avoid in available:
                seqs.append(LDBASequence([(reach, avoid)]))
        return seqs
    return wrapper


def chessworld_sample_reach_avoid_update(
        depth: int | tuple[int, int],
        num_reach: int | tuple[int, int],
        num_avoid: int | tuple[int, int],
        not_reach_same_as_last: bool = False
) -> Callable[[list[str]], LDBASequence]:
    def wrapper(propositions: list[str]) -> LDBASequence:
        def sample_one(last_reach, cur_d):
            # nr = random.randint(*num_reach) if isinstance(num_reach, tuple) else num_reach
            na = random.randint(*num_avoid) if isinstance(num_avoid, tuple) else num_avoid

            if cur_d == 0:
                na += 1

            not_in_reach = False

            mode = random.choice(['or', 'and', 'x_not_y'])

            if mode == 'and':
                available_reach = [a for a in all_ands if
                                   not last_reach.issubset(a)] if not_reach_same_as_last else all_ands
            else:

                available_reach = [a for a in complete_piece_assignments if
                                   not last_reach.issubset(a)] if not_reach_same_as_last else complete_piece_assignments

                if mode != "or":
                    not_in_reach = True

            reach = random.choice(available_reach)

            available_avoid = [a for a in complete_piece_assignments if (not last_reach.issubset(a)) or len(last_reach) == 0]
            try:
                avoid = frozenset.union(*random.sample(available_avoid, na)).difference(reach) if na > 0 else frozenset()
            except ValueError:
                print('Reach', reach)
                print('Last reach', last_reach)
                print('Available avoid', available_avoid)

            if not_in_reach:
                reach_minus = random.choice([x for x in complete_piece_assignments if not reach.issubset(x)])
                reach = reach - reach_minus

            return reach, avoid

        d = random.randint(*depth) if isinstance(depth, tuple) else depth
        last_reach = frozenset()
        seq = []
        for cur_d in range(d):
            reach, avoid = sample_one(last_reach, cur_d)
            seq.append((reach, avoid))
            last_reach = reach
        return LDBASequence(seq)

    return wrapper


def chessworld_sample_difficult_ra_update(depth: int | tuple[int, int]) -> Callable:
    def wrapper(propositions: list[str]) -> LDBASequence:
        d = random.randint(*depth) if isinstance(depth, tuple) else depth
        reach = random.choice(all_reach_difficult)
        avoid = complete_assignment.difference(reach)
        task = [(reach, avoid)]
        for _ in range(d - 1):
            reach = random.choice([a for a in all_reach if not reach.issubset(a)])
            task.append((reach, frozenset()))
        return LDBASequence(task)

    return wrapper


def chessworld_sample_reach_stay_update(num_stay: int, num_avoid: tuple[int, int]) -> Callable[[list[str]], LDBASequence]:
    def wrapper(propositions: list[str]) -> LDBASequence:
        mode = random.choice([1, 2, 3, 4])
        reach_minus = None

        if mode == 4:
            reach = random.choice(complete_piece_assignments)
            reach_minus = random.choice([a for a in complete_piece_assignments if not reach.issubset(a)])
        else:
            reach = random.choice(all_ands + all_ors)
        # while len(p.get_true_propositions()) > 1:
        #     p = random.choice(assignments)

        na = random.randint(*num_avoid)
        available = [a for a in all_assignments if a != reach and not reach.issubset(a)]
        avoid = random.sample([x for x in complete_piece_assignments if not reach.issubset(x)], na)
        avoid = frozenset.union(*avoid).difference(reach) if na > 0 else frozenset()
        second_avoid = frozenset.union(*all_assignments).difference(reach).union({Assignment.zero_propositions(propositions).to_frozen()})

        if reach_minus:
            reach = reach - reach_minus

        task = [(LDBASequence.EPSILON, avoid), (reach, second_avoid)]
        return LDBASequence(task, repeat_last=num_stay)

    return wrapper


if __name__ == '__main__':
    print(all_assignments)
