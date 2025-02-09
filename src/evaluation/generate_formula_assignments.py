from envs.chessworld import ChessWorld8
from itertools import combinations


def generate_formula_assignments(env, vars):
    all_assignments = env.get_possible_assignments()
    all_assignments_for_formula = []

    for assignment in all_assignments:
        active_props = assignment.get_true_propositions()

        if len(active_props) == 0:
            continue

        inactive_props = vars - active_props
        assignment_str = "&".join([prop for prop in list(active_props)] + ["!" + prop for prop in list(inactive_props)])

        # print(assignment_str)
        all_assignments_for_formula.append(assignment_str)

    return ["(" + assignment + ")" for assignment in all_assignments_for_formula]


def all_simple_reach_avoid(env, vars, goal, num_avoid):
    all_assignments = generate_formula_assignments(env, vars)
    all_avoid = list(combinations(all_assignments, num_avoid))

    # num_experiments = len(all_avoid)
    all_formulae = []

    for avoid in all_avoid:
        formula = "!(" + "|".join(avoid) + ") U " + goal
        all_formulae.append(formula)

    return all_formulae


if __name__ == "__main__":
    env = ChessWorld8()
    vars = set(env.PIECES.keys())
    formulaic_assignments = generate_formula_assignments(env, vars)

    print(formulaic_assignments)

    print(all_simple_reach_avoid(env, vars, "(rook & bishop)", 2))
