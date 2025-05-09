import itertools
import random
from pprint import pprint
from typing import Callable

from ltl.automata import LDBASequence
from ltl.logic import Assignment, FrozenAssignment
from envs import make_env
from envs.zones.quadrants import Quadrant
from ltl.samplers import AvoidSampler
import time

sampler = AvoidSampler.partial(depth=2, num_conjuncts=1)
zone_env = make_env('PointLtl2Debug-v0', sampler, render_mode='human', max_steps=2000)


props = set(zone_env.get_propositions())
assignments = zone_env.get_possible_assignments()
# assignments.remove(Assignment.zero_propositions(zone_env.get_propositions()))
all_assignments = [frozenset([a.to_frozen()]) for a in assignments]
complete_var_assignments = {}
colors_only = {}
areas_only = {}
opposites = {'right': 'left', 'left': 'right', 'top': 'bottom', 'bottom': 'top'}


agent_quadrant_to_assignment = {
    Quadrant.TOP_RIGHT: Assignment.where('right', 'top', propositions=props).to_frozen(),
    Quadrant.TOP_LEFT: Assignment.where('top', propositions=props).to_frozen(),
    Quadrant.BOTTOM_RIGHT: Assignment.where('right', propositions=props).to_frozen(),
    Quadrant.BOTTOM_LEFT: Assignment.zero_propositions(props).to_frozen()
}

# possible_avoids_from_location = {
#     ('right', ): ['left', 'left&top', 'bottom&left'],
#     ('left', ): ['right', 'right&top', 'bottom&right'],
#     ('bottom', ): ['top', 'left&top', 'right&top'],
#     ('top', ): ['bottom', 'bottom&left', 'bottom&right'],
#     ('right', 'top'): ['bottom', 'bottom&left', 'bottom&right', 'left', 'left&top'],
#     ('bottom', 'right'): ['left', 'bottom&left', 'right&top', 'top', 'left&top'],
#     ('bottom', 'left'): ['right', 'right&top', 'bottom&right', 'top', 'left&top'],
#     ('left', 'top'): ['bottom', 'bottom&left', 'bottom&right', 'right', 'right&top'],
#
# }


possible_avoids_from_location = {
    ('right', ): ['left'],
    ('left', ): ['right'],
    ('bottom', ): ['top'],
    ('top', ): ['bottom'],
    ('right', 'top'): ['bottom', 'left'],
    ('bottom', 'right'): ['left', 'top'],
    ('bottom', 'left'): ['right', 'top'],
    ('left', 'top'): ['bottom', 'right'],

}


always_reachable_assignments = {('top',), ('right',), (), ('right', 'top')}

complete_assignment = frozenset.union(*all_assignments)

sample_voc = {0: 'PAD', 1: 'EPSILON', 2: 'NULL', 3: 'blue', 4: 'green', 5: 'magenta', 6: 'yellow', 7: 'right', 8: 'top',
              9: 'right&blue', 10: 'right&green', 11: 'right&magenta', 12: 'right&yellow', 13: 'top&blue',
              14: 'top&green', 15: 'top&magenta', 16: 'top&yellow', 17: 'right&top', 18: 'right&top&blue',
              19: 'right&top&green', 20: 'right&top&magenta', 21: 'right&top&yellow', 22: 'blank'}


def get_complete_var_assignments(cur_var: str) -> frozenset[FrozenAssignment]:
    var_assignments = []
    for assignment in assignments:
        if cur_var == 'left':
            if 'right' not in assignment.get_true_propositions():
                var_assignments.append(assignment.to_frozen())
        elif cur_var == 'bottom':
            if 'top' not in assignment.get_true_propositions():
                var_assignments.append(assignment.to_frozen())
        else:
            if cur_var in assignment.get_true_propositions():
                var_assignments.append(assignment.to_frozen())
    return frozenset(var_assignments)


for variable in ['blue', 'green', 'magenta', 'yellow', 'right', 'top', 'left', 'bottom']:
    cur_var_assignment = get_complete_var_assignments(variable)
    all_assignments.append(cur_var_assignment)
    complete_var_assignments[variable] = cur_var_assignment

    if variable in ['blue', 'green', 'magenta', 'yellow']:
        colors_only[variable] = cur_var_assignment
    else:
        areas_only[variable] = cur_var_assignment


reach_ors = {i: [frozenset.union(*x) for x in itertools.combinations(list(complete_var_assignments.values()), i)
                 if frozenset.union(*x) != complete_assignment] for i in range(1, 3)}
# all_ors = reach_ors[1] + reach_ors[2]
all_ors = reach_ors[1]

all_ands = []
all_ands_dict = {}

# originally (2, 4)
for i in range(2, 3):
    for key_tup in itertools.combinations(list(complete_var_assignments.keys()), i):
        tup = [complete_var_assignments[cur_key] for cur_key in key_tup]
        and_set = frozenset.intersection(*tup)

        if len(and_set) > 0 and and_set not in all_ands:
            all_ands.append(and_set)
            all_ands_dict['&'.join(sorted(key_tup))] = and_set

# print(len(all_ands))
# print(all_ands)

# all_pairs = itertools.combinations(list(complete_var_assignments.values()), 2)

all_pairs = [(x, y) for x in areas_only.values() for y in colors_only.values()]
# reach_x_and_not_y = [x - y for x, y in all_pairs if len(x & y) > 0] + [y - x for x, y in all_pairs if len(x & y) > 0]
reach_x_and_not_y = [x - y for x, y in all_pairs if len(x & y) > 0]

all_reach = all_ands + reach_x_and_not_y + all_ors

# all_reach_difficult = all_ors + all_ands

# print(len(all_reach))


def check_feasible_reach(task: frozenset[FrozenAssignment], info_dict):

    def fix_label(assignment_label):
        return tuple(sorted([i.strip() for i in assignment_label.split("&") if (len(i) > 0 and '!' not in i)]))

    task_labs = set([fix_label(fr.to_label()) for fr in task])
    # print(task_labs)
    # print([fr.to_label() for fr in task])

    def check_feasible_assignment(assignment_list):
        # print('nontrivial')
        assignment_set = set(assignment_list)

        color = assignment_set & {'blue', 'green', 'magenta', 'yellow'}
        rest = assignment_set - color

        color_id = list(color)[0]
        relevant_quadrants = info_dict[color_id]

        if len(assignment_set) == 1:
            return Quadrant.BOTTOM_LEFT in relevant_quadrants

        elif rest == {'right'}:
            return Quadrant.BOTTOM_RIGHT in relevant_quadrants

        elif rest == {'top'}:
            return Quadrant.TOP_LEFT in relevant_quadrants

        else:
            return Quadrant.TOP_RIGHT in relevant_quadrants

    if len(task_labs & always_reachable_assignments) > 0:
        return True

    return any(check_feasible_assignment(task_label) for task_label in task_labs)


def check_nontrivial(agent_quadrant, reach, depth):
    if depth > 0:
        return True

    trivializing_assignment = agent_quadrant_to_assignment[agent_quadrant]
    # trivial_set = frozenset([trivializing_assignment])

    # if trivializing_assignment in reach:
    #     print(trivializing_assignment, reach, trivial_set & reach)

    return not (trivializing_assignment in reach)


def agent_and_color_sample(agent_quadrant, color_quadrant):
    quadrant_to_keys = {
        (Quadrant.TOP_LEFT, Quadrant.TOP_LEFT): [random.choice(['bottom', 'right'])],
        (Quadrant.TOP_RIGHT, Quadrant.TOP_RIGHT): [random.choice(['bottom', 'left'])],
        (Quadrant.BOTTOM_LEFT, Quadrant.BOTTOM_LEFT): [random.choice(['top', 'right'])],
        (Quadrant.BOTTOM_RIGHT, Quadrant.BOTTOM_RIGHT): [random.choice(['left', 'top'])],
        (Quadrant.TOP_LEFT, Quadrant.TOP_RIGHT): ['bottom'],
        (Quadrant.TOP_LEFT, Quadrant.BOTTOM_LEFT): ['right'],
        (Quadrant.TOP_LEFT, Quadrant.BOTTOM_RIGHT): [random.choice(['right&top', 'bottom&left'])],
        (Quadrant.TOP_RIGHT, Quadrant.BOTTOM_LEFT): [random.choice(['left&top', 'bottom&right'])],
        (Quadrant.TOP_RIGHT, Quadrant.BOTTOM_RIGHT): ['left'],
        (Quadrant.BOTTOM_LEFT, Quadrant.BOTTOM_RIGHT): ['top'],
    }

    try:
        return quadrant_to_keys[(agent_quadrant, color_quadrant)]

    except KeyError:

        return quadrant_to_keys[(color_quadrant, agent_quadrant)]


def zonenv_all_reach_tasks(depth: int) -> Callable:
    def wrapper(propositions: list[str], info_dict: dict[str, list[Quadrant]]) -> list[LDBASequence]:
        reachs = all_reach

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


def zonenv_sample_reach(depth: int | tuple[int, int]) -> Callable:
    def wrapper(propositions: list[str], info_dict: dict[str, list[Quadrant]]) -> LDBASequence:
        # start = time.time()
        d = random.randint(*depth) if isinstance(depth, tuple) else depth
        # reach = random.choice(all_reach)
        # task = [(reach, frozenset())]

        reach = None
        task = []

        agent_quadrant = info_dict['agent']
        for cur_d in range(d):

            if cur_d == 0:
                possible_reach = [a for a in all_reach]
            else:
                possible_reach = [a for a in all_reach if (not reach.issubset(a))]

            reach = random.choice(possible_reach)

            while not (check_feasible_reach(reach, info_dict) and check_nontrivial(agent_quadrant, reach, cur_d)):
                reach = random.choice(possible_reach)
            # reach = random.choice([a for a in all_reach if (not reach.issubset(a)) ])
            task.append((reach, frozenset()))

        # end = time.time()
        # print(end-start)
        return LDBASequence(task)

    return wrapper


# TODO fix rest
def zonenv_all_reach_avoid():
    def wrapper(_):
        seqs = []
        for reach in all_assignments:
            available = [a for a in all_assignments if a != reach and not reach.issubset(a)]
            for avoid in available:
                seqs.append(LDBASequence([(reach, avoid)]))
        return seqs
    return wrapper


def zonenv_sample_reach_avoid(
        depth: int | tuple[int, int],
        num_reach: int | tuple[int, int],
        num_avoid: int | tuple[int, int],
        not_reach_same_as_last: bool = False
) -> Callable[[list[str]], LDBASequence]:
    def wrapper(propositions: list[str], info_dict: dict[str, list[Quadrant]]) -> LDBASequence:

        def sample_one_color(last_reach, cur_d):
            # nr = random.randint(*num_reach) if isinstance(num_reach, tuple) else num_reach
            na = random.randint(*num_avoid) if isinstance(num_avoid, tuple) else num_avoid

            # if cur_d == 0:
            #     na += 1

            not_in_reach = False

            # mode = random.choice(['or', 'and', 'x_not_y'])
            mode = random.choice(['or', 'and'])

            ra_encoding = []
            agent_quadrant = info_dict['agent']

            if mode == 'and':
                available_reach = [k for k, a in all_ands_dict.items() if
                                   (not last_reach.issubset(a)) and check_feasible_reach(a, info_dict) and check_nontrivial(agent_quadrant, a, cur_d)] if not_reach_same_as_last else [k for k, a in all_ands_dict.items()
                                                                                                                                        if (check_feasible_reach(a, info_dict) and check_nontrivial(agent_quadrant, a, cur_d))]

                reach_key = random.choice(available_reach)
                ra_encoding.append(reach_key)

                reach = all_ands_dict[reach_key]

            else:

                available_reach = [k for k, a in complete_var_assignments.items() if
                                   (not last_reach.issubset(a)) and check_nontrivial(agent_quadrant, a, cur_d)] if not_reach_same_as_last else [k for k, a in complete_var_assignments.items() if check_nontrivial(agent_quadrant, a, cur_d)]

                reach_key = random.choice(available_reach)
                ra_encoding.append(reach_key)

                reach = complete_var_assignments[reach_key]

                if mode != "or":
                    not_in_reach = True

            # reach = random.choice(available_reach)

            if na > 0:
                available_avoid = [k for k, a in colors_only.items() if (not last_reach.issubset(a)) or len(last_reach) == 0]
                avoid_keys = random.sample(available_avoid, na)

                ra_encoding.append(set(avoid_keys) - {reach_key})

                avoid = frozenset.union(*[colors_only[avoid_key] for avoid_key in avoid_keys]).difference(reach)
            else:
                ra_encoding.append(set([]))
                avoid_keys = None
                avoid = frozenset()

            # try:
            #     avoid = frozenset.union(*random.sample(available_avoid, na)).difference(reach) if na > 0 else frozenset()
            # except ValueError:
            #     print('Reach', reach)
            #     print('Last reach', last_reach)
            #     print('Available avoid', available_avoid)

            if not_in_reach:
                reach_minus_list = [k for k, x in areas_only.items() if (not reach.issubset(x))
                                    and check_feasible_reach(reach - x, info_dict)]

                if len(reach_minus_list) > 0:

                    reach_minus_key = random.choice(reach_minus_list)
                    reach_minus = complete_var_assignments[reach_minus_key]
                    reach = reach - reach_minus

                    ra_encoding.append(reach_minus_key)
                else:
                    ra_encoding.append(None)

            else:
                ra_encoding.append(None)

            if len(avoid) > 10:
                print("long avoid color")
                print(avoid_keys)
                print(reach_key)

            return reach, avoid, ra_encoding

        def sample_one_area(last_reach, last_ra_encoding):
            na = random.randint(*num_avoid) if isinstance(num_avoid, tuple) else num_avoid

            ra_encoding = []

            def retrieve_areas(area_key):
                try:
                    return areas_only[area_key]
                except KeyError:
                    return all_ands_dict[area_key]

            if na == 0:
                reach_key = random.choice(list(colors_only.keys()))
                reach = colors_only[reach_key]

                ra_encoding = [reach_key, set([]), None]

                return reach, frozenset(), ra_encoding

            if not last_ra_encoding:
                # print("Startin mate")
                agent_quadrant = info_dict['agent']

                reach_key = random.choice(list(colors_only.keys()))
                color_quadrant = random.choice(list(info_dict[reach_key]))

                avoid_keys_areas = agent_and_color_sample(agent_quadrant, color_quadrant)

                if na > 1 and '&' not in avoid_keys_areas[0]:
                    avoid_keys_colors = random.sample(list(set(colors_only.keys()) - {reach_key}), na-1)
                else:
                    avoid_keys_colors = []

                avoid_assignment_sets = [retrieve_areas(ak) for ak in avoid_keys_areas] + [colors_only[ak] for ak in avoid_keys_colors]
                avoid_keys = avoid_keys_areas + avoid_keys_colors

                reach = colors_only[reach_key]
                avoid = frozenset.union(*avoid_assignment_sets).difference(reach)

                ra_encoding.append(reach_key)
                ra_encoding.append(set(avoid_keys))

            elif any(prop in opposites for prop in last_ra_encoding[0].split("&")):
                # print("E")
                # print(last_ra_encoding)
                last_reach_props = last_ra_encoding[0].split("&")

                possible_avoids = possible_avoids_from_location[tuple(sorted([x for x in last_reach_props if x in opposites]))]

                possible_reach = [k for k, a in colors_only.items() if
                                  not last_reach.issubset(a)] if not_reach_same_as_last else list(colors_only.keys())
                reach_key = random.choice(possible_reach)
                reach = colors_only[reach_key]

                cur_possible_avoids = list(filter(lambda x: check_feasible_reach(reach.difference(retrieve_areas(x)),
                                                                             info_dict), possible_avoids))

                prev = {reach_key}
                while len(cur_possible_avoids) == 0:
                    sampling_reach_set = set(possible_reach) - prev

                    if len(sampling_reach_set) == 0:
                        print("warning: difficult sampling")
                        print(last_reach_props, possible_avoids_from_location[
                            tuple(sorted([x for x in last_reach_props if x in opposites]))])
                        print(info_dict)
                        ra_encoding.append(reach_key)
                        ra_encoding.append(set([]))
                        ra_encoding.append(None)

                        return reach, frozenset(), ra_encoding

                    reach_key = random.choice(list(sampling_reach_set))

                    prev.add(reach_key)
                    reach = colors_only[reach_key]

                    cur_possible_avoids = list(filter(lambda x: check_feasible_reach(
                        reach.difference(retrieve_areas(x)),
                        info_dict), possible_avoids))

                avoid_key_area = random.choice(cur_possible_avoids)
                avoid_color_keys = []

                if na > 1:
                    possible_avoid_colors = [x for x, a in colors_only.items() if not last_reach.issubset(a)]
                    avoid_color_keys = random.sample(possible_avoid_colors, na-1)

                all_avoid_sets = ([retrieve_areas(avoid_key_area)]
                                  + [colors_only[ck] for ck in avoid_color_keys])
                avoid = frozenset.union(*all_avoid_sets).difference(reach)

                avoid_keys = set([avoid_key_area] + avoid_color_keys) - {reach_key}


                ra_encoding.append(reach_key)
                ra_encoding.append(avoid_keys)

            else:
                """If !color U color choose a quadrant of the color as starting position for the agent (diff task)
                and sample as before (random other color and quadrant of color)
                
                If !area U color, check which possible color quadrants are possible first and then do as above
                
                Should be able to handle both at once: let prev_c be the last color. If info_dic"""

                prev_c = last_ra_encoding[0]

                if len(info_dict[prev_c]) == 1:
                    agent_quadrant = list(info_dict[prev_c])[0]

                else:
                    def reconstruct_assignment_set(proposition_encodings):
                        assignment_set = []

                        for prop in proposition_encodings:
                            try:
                                assignment_set.append(complete_var_assignments[prop])
                            except KeyError:
                                assignment_set.append(all_ands_dict[prop])

                        if len(assignment_set) == 0:
                            return frozenset()

                        return frozenset.union(*assignment_set)


                    agent_quadrant = None
                    prev_task = last_reach.difference(reconstruct_assignment_set(last_ra_encoding[1]))

                    for possible_agent_quadrant in list(info_dict[prev_c]):
                        new_info_dict = {k: v for k, v in info_dict.items()}
                        new_info_dict[prev_c] = {possible_agent_quadrant}

                        if check_feasible_reach(prev_task, new_info_dict):
                            agent_quadrant = possible_agent_quadrant
                            break

                    if not agent_quadrant:
                        print(last_ra_encoding)
                        print(info_dict)
                        raise AssertionError('Prev task impossible')

                reach_key = random.choice(list(colors_only.keys()))
                color_quadrant = random.choice(list(info_dict[reach_key]))

                avoid_keys_areas = agent_and_color_sample(agent_quadrant, color_quadrant)

                if na > 1 and '&' not in avoid_keys_areas[0]:
                    avoid_keys_colors = random.sample([x for x, a in colors_only.items() if not last_reach.issubset(a)], na - 1)
                else:
                    avoid_keys_colors = []

                avoid_assignment_sets = [retrieve_areas(ak) for ak in avoid_keys_areas] + [colors_only[ak] for ak in
                                                                                       avoid_keys_colors]
                avoid_keys = avoid_keys_areas + avoid_keys_colors

                reach = colors_only[reach_key]
                avoid = frozenset.union(*avoid_assignment_sets).difference(reach)

                ra_encoding.append(reach_key)
                ra_encoding.append(set(avoid_keys))

            ra_encoding.append(None)

            if len(avoid) > 10:
                print('Long avoid area')
                print(avoid_keys)
                print(reach_key)

            # print(ra_encoding, info_dict)

            return reach, avoid, ra_encoding

        d = random.randint(*depth) if isinstance(depth, tuple) else depth
        last_reach = frozenset()
        last_ra_encoding = None
        mode = 0

        seq = []
        for cur_d in range(d):
            # if mode == 0:
            #     mode = random.choice([0, 1])
            # else:
            #     mode = 0

            mode = random.choice([0, 1])

            if mode == 0:
                reach, avoid, ra_encoding = sample_one_color(last_reach, cur_d)
            else:
                reach, avoid, ra_encoding = sample_one_area(last_reach, last_ra_encoding)

            seq.append((reach, avoid))
            last_reach = reach
            last_ra_encoding = ra_encoding
        #     print(last_ra_encoding)
        # print(info_dict)
        return LDBASequence(seq)

    return wrapper


# def zonenv_sample_difficult_ra_update(depth: int | tuple[int, int]) -> Callable:
#     def wrapper(propositions: list[str], info_dict: dict[str, list[Quadrant]]) -> LDBASequence:
#         d = random.randint(*depth) if isinstance(depth, tuple) else depth
#         reach = random.choice(all_reach_difficult)
#         avoid = complete_assignment.difference(reach)
#         task = [(reach, avoid)]
#         for _ in range(d - 1):
#             reach = random.choice([a for a in all_reach if not reach.issubset(a)])
#             task.append((reach, frozenset()))
#         return LDBASequence(task)
#
#     return wrapper
#
#
# def zonenv_sample_reach_stay_update(num_stay: int, num_avoid: tuple[int, int]) -> Callable[[list[str]], LDBASequence]:
#     def wrapper(propositions: list[str], info_dict: dict[str, list[Quadrant]]) -> LDBASequence:
#         mode = random.choice([1, 2, 3, 4])
#         reach_minus = None
#
#         if mode == 4:
#             reach = random.choice(complete_var_assignments)
#             reach_minus = random.choice([a for a in complete_var_assignments if not reach.issubset(a)])
#         else:
#             reach = random.choice(all_ands + all_ors)
#         # while len(p.get_true_propositions()) > 1:
#         #     p = random.choice(assignments)
#
#         na = random.randint(*num_avoid)
#         available = [a for a in all_assignments if a != reach and not reach.issubset(a)]
#         avoid = random.sample([x for x in complete_var_assignments if not reach.issubset(x)], na)
#         avoid = frozenset.union(*avoid).difference(reach) if na > 0 else frozenset()
#         second_avoid = frozenset.union(*all_assignments).difference(reach).union({Assignment.zero_propositions(propositions).to_frozen()})
#
#         if reach_minus:
#             reach = reach - reach_minus
#
#         task = [(LDBASequence.EPSILON, avoid), (reach, second_avoid)]
#         return LDBASequence(task, repeat_last=num_stay)
#
#     return wrapper


if __name__ == '__main__':
    # print(all_assignments)

    sample_info_dict = {'blue': {Quadrant.TOP_RIGHT, Quadrant.TOP_LEFT}, 'green': {Quadrant.BOTTOM_LEFT},
                        'magenta': {Quadrant.BOTTOM_RIGHT, Quadrant.TOP_LEFT}, 'yellow': {Quadrant.BOTTOM_LEFT, Quadrant.TOP_LEFT}}



    print(all_ands_dict['green&top'])
    print(check_feasible_reach(all_ands_dict['green&top'], sample_info_dict))

    always_possible = ['blue', 'green', 'magenta', 'yellow', 'right', 'top', 'left', 'bottom', 'right&top',
                       'bottom&right', 'bottom&left', 'left&top']

    for test_task in always_possible:
        print(test_task)
        if '&' not in test_task:
            assert(check_feasible_reach(complete_var_assignments[test_task], sample_info_dict))
        else:
            # print(all_ands_dict[test_task])
            assert(check_feasible_reach(all_ands_dict[test_task], sample_info_dict))

    print("TEST")
    for _ in range(5):
        print(agent_and_color_sample(Quadrant.BOTTOM_LEFT, Quadrant.TOP_RIGHT))


    print('Lengths')
    print(len(all_reach))
    print(len(all_ors))
    print(len(all_ands))
    print(len(reach_x_and_not_y))


    def retrieve_areas_test(area_key):
        try:
            return areas_only[area_key]
        except KeyError:
            return all_ands_dict[area_key]

    sample_info_dict_2 = {'green': {Quadrant.TOP_RIGHT}, 'magenta': {Quadrant.BOTTOM_RIGHT, Quadrant.TOP_RIGHT},
                        'blue': {Quadrant.BOTTOM_LEFT, Quadrant.BOTTOM_RIGHT},
                        'yellow': {Quadrant.BOTTOM_LEFT, Quadrant.TOP_LEFT}, 'agent': Quadrant.TOP_LEFT}

    for c, a in colors_only.items():
        r = retrieve_areas_test('right')

        print(check_feasible_reach(a.difference(r), sample_info_dict_2))

    print(all_ands_dict['bottom&left'])
    # print(tuple(sorted([i.strip() for i in all_ands_dict['bottom&left'].to_label().split("&") if (len(i) > 0 and '!' not in i)])))