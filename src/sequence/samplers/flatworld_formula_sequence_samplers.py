from envs.flatworld import FlatWorld
from model.formulae_utils.ContextMaker import ContextMaker
import random
from pprint import pprint
from typing import Callable

from ltl.automata import LDBASequence
from ltl.logic import Assignment, FrozenAssignment

sample_vocab = {0: 'PAD', 1: 'EPSILON', 2: 'NULL', 3: 'red', 4: 'magenta', 5: 'red&magenta', 6: 'blue', 7: 'green',
                8: 'aqua', 9: 'green&blue', 10: 'green&aqua', 11: 'aqua&blue', 12: 'green&aqua&blue', 13: 'yellow',
                14: 'orange', 15: 'blank'}

var_names = ['aqua', 'blue', 'green', 'magenta', 'orange', 'red', 'yellow', 'EPSILON', 'NULL', 'blank']
true_vars = ['aqua', 'blue', 'green', 'magenta', 'orange', 'red', 'yellow']

context_maker = ContextMaker(sample_vocab, var_names, true_vars)
context_maker.generate_cache()

organized_formulae = context_maker.formula_kinds
organized_formulae_blank = context_maker.blank_formula_kinds

flatworld = FlatWorld()
props = set(flatworld.get_propositions())
assignments = flatworld.get_possible_assignments()

actual_assignments_vocab = {0: 'PAD', 1: 'EPSILON', 2: 'NULL'}

for i, assignment_name in sample_vocab.items():
    for assignment in assignments:
        assignment_set = set([piece for piece in assignment if assignment[piece]])

        if assignment_set == set(assignment_name.split("&")):
            actual_assignments_vocab[i] = assignment
        elif len(assignment_set) == 0 and assignment_name == "blank":
            actual_assignments_vocab[i] = assignment

reverse_assignment_vocab = {val if isinstance(val, str) else val.to_frozen(): key for key, val in actual_assignments_vocab.items()}


relevant_formulae_simple = [i for i in organized_formulae["or"]["positive"][2]]
# relevant_formulae_simple += [i for i in organized_formulae["or"]["negative"][2]]

relevant_formulae_simple = [i for i in organized_formulae["or"]["positive"][1]]
# relevant_formulae_simple += [i for i in organized_formulae["or"]["negative"][1]]

relevant_formulae_simple += [i for i in organized_formulae["and"]["positive"][2]]
relevant_formulae_simple += [i for i in organized_formulae["and"]["positive"][3]]
# relevant_formulae_simple += [i for i in organized_formulae["and"]["negative"][2]]

final_relevant_formulae_simple = [frozenset([actual_assignments_vocab[i].to_frozen() for i in tup])
                                   for tup in relevant_formulae_simple]

relevant_formulae_complex = [i for i in organized_formulae["or_x_and_y"]["positive"][(2, 1)]]
# relevant_formulae_complex += [i for i in organized_formulae["or_x_and_y"]["negative"][(2, 1)]]

relevant_formulae_complex += [i for i in organized_formulae["and_x_and_not_y"]["positive"][(2, 1)]]
# relevant_formulae_complex += [i for i in organized_formulae["and_x_and_not_y"]["negative"][(2, 1)]]


for i in [1, 2]:
    for j in [1, 2]:
        relevant_formulae_complex += organized_formulae["or_x_and_not_y"]["positive"][(i, j)]
        # relevant_formulae_complex += organized_formulae["or_x_and_not_y"]["negative"][(i, j)]

final_relevant_formulae_complex = [frozenset([actual_assignments_vocab[i].to_frozen() for i in tup])
                                   for tup in relevant_formulae_complex]

piece_avoid_dataset = {}

for num_reach in range(1, 3):
    piece_avoid_dataset[num_reach] = {}

    for piece_set in organized_formulae["or"]["positive"][num_reach]:
        piece_avoid_dataset[num_reach][piece_set] = {}
        piece_avoid_dataset[num_reach][piece_set][piece_set] = frozenset([actual_assignments_vocab[i].to_frozen()
                                                                          for i in piece_set])

        piece_assignment_set = set(piece_set)

        for num_avoid in range(1, 3):
            possible_nice_avoids = [x for x in organized_formulae["or_x_and_not_y"]["positive"][(num_avoid, num_reach)]
                                    if len(set(x) & piece_assignment_set) == 0]

            possible_nice_avoids += [x for x in organized_formulae["or"]["positive"][num_avoid]
                                     if len(set(x) & piece_assignment_set) == 0]

            possible_nice_avoids_assignments = [frozenset([actual_assignments_vocab[i].to_frozen() for i in x])
                                                for x in possible_nice_avoids]

            piece_avoid_dataset[num_reach][piece_set][num_avoid] = possible_nice_avoids_assignments


ands_avoid_dataset = {}
for num_ands in range(1, 3):
    ands_avoid_dataset[num_ands] = {}

    if num_ands == 1:
        cur_formula_data = organized_formulae["and"]["positive"][2] + organized_formulae["and"]["positive"][3]
    else:
        cur_formula_data = organized_formulae["or_x_and_y"]["positive"][(num_ands, 1)]

    # print("Num ands ", num_ands)

    for and_set in cur_formula_data:
        ands_avoid_dataset[num_ands][and_set] = {}
        ands_avoid_dataset[num_ands][and_set][and_set] = frozenset([actual_assignments_vocab[i].to_frozen()
                                                                     for i in and_set])

        and_assignment_set = set(and_set)

        for num_avoid in range(1, 3):

            if num_avoid == 1:
                possible_sets = organized_formulae["or_x_and_not_y"]["positive"][(num_avoid, 1)]

            else:

                possible_sets = [x for x in organized_formulae["or_x_and_not_ny"]["positive"][(num_avoid, 2, num_ands)]]

            possible_sets += [x for x in organized_formulae["or"]["positive"][num_avoid]]

            # possible_sets += [x for x in organized_formulae["or_x_and_not_y"][(num_avoid, 1)]

            possible_nice_avoids = [x for x in possible_sets
                                    if len(set(x) & and_assignment_set) == 0]

            try:
                assert (len(possible_nice_avoids) > 0)

            except AssertionError:
                print("Can't sample for this")
                print(and_assignment_set, num_avoid)

            possible_nice_avoids_assignments = [frozenset([actual_assignments_vocab[i].to_frozen() for i in x])
                                                for x in possible_nice_avoids]

            ands_avoid_dataset[num_ands][and_set][num_avoid] = possible_nice_avoids_assignments


reach_stay_assignments = [i for i in organized_formulae["or"]["positive"][1]]
reach_stay_assignments += [i for i in organized_formulae["or"]["positive"][2]]
# reach_stay_assignments += [i for i in organized_formulae["and"]["positive"][2]]
# reach_stay_assignments += [i for i in organized_formulae["or_x_and_y"]["positive"][(2, 1)]]

all_ands_formulae = organized_formulae["and"]["positive"][2] + organized_formulae["and"]["positive"][3]


def flatworld_sample_simple_reach(depth: int | tuple[int, int], num_ors: int | tuple[int, int]) -> Callable:
    def wrapper(propositions: list[str]) -> LDBASequence:
        d = random.randint(*depth) if isinstance(depth, tuple) else depth

        reach = random.choice(final_relevant_formulae_simple)

        task = [(reach, frozenset())]
        for _ in range(d - 1):
            reach = random.choice([a for a in final_relevant_formulae_simple if not reach.issubset(a)])
            task.append((reach, frozenset()))
        return LDBASequence(task)

    return wrapper


def flatworld_sample_complex_reach(depth: int | tuple[int, int]) -> Callable:
    def wrapper(propositions: list[str]) -> LDBASequence:
        d = random.randint(*depth) if isinstance(depth, tuple) else depth

        reach = random.choice(final_relevant_formulae_complex)

        task = [(reach, frozenset())]
        for _ in range(d - 1):
            reach = random.choice([a for a in final_relevant_formulae_complex if not reach.issubset(a)])
            task.append((reach, frozenset()))
        return LDBASequence(task)

    return wrapper


def flatworld_sample_formulae_reach_avoid(
        depth: int | tuple[int, int],
        num_reach: int | tuple[int, int],
        num_avoid: int | tuple[int, int],
        not_reach_same_as_last: bool = False
) -> Callable[[list[str]], LDBASequence]:
    def wrapper(propositions: list[str]) -> LDBASequence:
        def sample_one(last_reach, seq):

            """TODO:
               for each depth, sample (and/or) first
               then sample num_ors/num_ands
               sample the desired reach, then the desired avoid from the dataset
               do disjoint tasks as below
            """
            nr = random.randint(*num_reach) if isinstance(num_reach, tuple) else num_reach

            if nr > 1:
                na = 1
            else:
                na = random.randint(*num_avoid) if isinstance(num_avoid, tuple) else num_avoid

            and_mode = random.choice([True, False])

            if and_mode:
                if nr == 1:
                    all_assignments = all_ands_formulae
                else:
                    all_assignments = organized_formulae["or_x_and_y"]["positive"][(nr, 1)]

                dataset = ands_avoid_dataset
            else:
                all_assignments = organized_formulae["or"]["positive"][nr]
                dataset = piece_avoid_dataset

            available = [a for a in all_assignments if not set(a).issubset(set(last_reach))] if not_reach_same_as_last else all_assignments

            assert (len(available) > 0)

            available_avoid = []
            c = 0

            while len(available_avoid) == 0 and c < 10:
                reach = random.choice(available)
                reach_assignment = dataset[nr][reach][reach]

                available_avoid = dataset[nr][reach][na]

                if len(list(last_reach)) > 0:
                    available_avoid = [x for x in available_avoid if not last_reach.issubset(x)]

                c += 1

            if len(available_avoid) == 0:
                print("Last reach", last_reach)
                print("seq", seq)
                print("Reach", reach)
                print("C", c)
                print("Num reach", nr)
                print("Num avoid", na)
                print("Available", available)
                raise AssertionError("Can't samle this")


            avoid_assignment = random.choice(available_avoid)

            return reach_assignment, avoid_assignment

        d = random.randint(*depth) if isinstance(depth, tuple) else depth
        last_reach = frozenset([])
        seq = []
        for _ in range(d):
            reach, avoid = sample_one(last_reach, seq)

            seq.append((reach, avoid))
            last_reach = reach

        return LDBASequence(seq)

    return wrapper


def flatworld_sample_formula_reach_stay(num_stay: int, num_avoid: tuple[int, int]) -> Callable[[list[str]], LDBASequence]:
    def wrapper(propositions: list[str]) -> LDBASequence:
        available = []
        c = 0

        while len(available) == 0 or c < 10:
            nr = random.choice([1, 2])
            # nr = 1

            if nr == 3:
                reach_tup = random.choice(all_ands_formulae)
            else:
                reach_tup = random.choice(organized_formulae["or"]["positive"][nr])

            reach = frozenset([actual_assignments_vocab[i].to_frozen() for i in reach_tup])
            # while len(p.get_true_propositions()) > 1:
            #     p = random.choice(assignments)
            na = random.randint(*num_avoid)

            if na == 0:
                available = [frozenset()]

            else:
                available = piece_avoid_dataset[nr][reach_tup][na]

            c += 1

        if len(available) == 0:
            raise AssertionError("Can't samle this")

        avoid = random.choice(available)
        # avoid = frozenset.union(*avoid) if len(avoid) > 0 else frozenset()

        complete_assignment = frozenset([actual_assignments_vocab[i].to_frozen() for i in context_maker.complete_assignment])
        second_avoid = complete_assignment.difference(reach).union({Assignment.zero_propositions(propositions).to_frozen()})
        task = [(LDBASequence.EPSILON, avoid), (reach, second_avoid)]
        return LDBASequence(task, repeat_last=num_stay)

    return wrapper


def count_all_formulae():
    tot = 0
    tot += len(relevant_formulae_complex)**2
    tot += len(relevant_formulae_simple)**2
    tot += len(relevant_formulae_simple)
    tot += len(relevant_formulae_complex)

    datasets = [piece_avoid_dataset, ands_avoid_dataset]
    for nr_1 in [1]:
        if nr_1 == 1:
            all_assignments = organized_formulae["and"]["positive"][2]
        else:
            all_assignments = organized_formulae["or_x_and_y"]["positive"][(nr_1, 1)]
        all_assignments = [(1, x) for x in all_assignments]

        all_assignments += [(0, x) for x in organized_formulae["or"]["positive"][nr_1]]

        for i, last_reach in all_assignments:
            # reach_assignment = datasets[i][nr_1][last_reach][last_reach]
            lra = datasets[i][nr_1][last_reach][last_reach]
            for na_1 in range(1, 4 - nr_1):

                available_avoid = datasets[i][nr_1][last_reach][na_1]
                tot += len(available_avoid)

                for nr_2 in [1]:
                    if nr_2 == 1:
                        all_assignments_2 = organized_formulae["and"]["positive"][2]
                    else:
                        all_assignments_2 = organized_formulae["or_x_and_y"]["positive"][(nr_2, 1)]
                    all_assignments_2 = [(1, x) for x in all_assignments_2]

                    all_assignments_2 += [(0, x) for x in organized_formulae["or"]["positive"][nr_2]]

                    for j, reach in all_assignments_2:
                        for na_2 in range(1, 4 - nr_2):
                            available_avoid_2 = datasets[j][nr_2][reach][na_2]

                            if len(list(last_reach)) > 0:
                                available_avoid_2 = [x for x in available_avoid_2 if not lra.issubset(x)]

                            tot += len(available_avoid) * len(available_avoid_2)

    return tot


if __name__ == "__main__":
    print(actual_assignments_vocab)
    print(organized_formulae["or_x_and_y"]["positive"][(2, 1)])

    for num_reach, d in piece_avoid_dataset.items():
        print("Num reach", num_reach)

        for key, d1 in d.items():
            print("Piece", key)

            for k, v in d1.items():
                print(k, v)
    print(ands_avoid_dataset)

    for piece, dk in piece_avoid_dataset[2].items():
        print(piece, dk)

    print(len(final_relevant_formulae_simple))
    print(len(final_relevant_formulae_complex))
    print(len(organized_formulae["or_x_and_not_ny"]["positive"][(2, 2, 1)]))
    print(len(organized_formulae["or_x_and_not_ny"]["positive"][(2, 2, 2)]))

    print(count_all_formulae())

    for key, val in context_maker.cache.items():
        print(key, val)

    print(context_maker.cache[(4, 5, 7, 8, 10, 11, 13)])