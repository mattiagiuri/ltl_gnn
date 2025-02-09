from itertools import product
# from sequence.samplers.chessworld_sequence_samplers import chessworld_sample_reach_avoid, chessworld_all_reach_avoid
# from envs.chessworld.chessworld import ChessWorld


class ChessEnv():
    PIECES = {
        "queen": [(7, 0)],
        "rook": [(7, 3)],
        "knight": [(3, 4)],
        "bishop": [(1, 4), (7, 5)],
        "pawn": [(4, 3)]
    }

    ATTACKED_SQUARES = {
        "queen": {(7, 0), (7, 1), (7, 2), (7, 3), (6, 1), (5, 2), (4, 3),
                  (6, 0), (5, 0), (4, 0), (3, 0)},
        "rook": {(7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
                 (6, 3), (5, 3), (4, 3)},
        "knight": {(3, 4), (4, 2), (5, 3), (5, 5), (4, 6),
                   (2, 6), (1, 5), (1, 3), (2, 2)},
        "bishop": {(3, 0), (4, 1), (5, 2), (6, 3), (7, 4), (2, 1), (1, 2), (0, 3),
                   (5, 7), (6, 6), (7, 5), (4, 6), (3, 5), (2, 4), (1, 3), (0, 2)},
        "pawn": {(4, 3), (5, 2), (5, 4)}

    }

    FREE_SQUARES = {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (3, 1), (5, 1),
                    (3, 2), (6, 2), (2, 3), (3, 3), (0, 4), (1, 4), (4, 4), (6, 4),
                    (0, 5), (2, 5), (4, 5), (6, 5), (0, 6), (1, 6), (3, 6), (5, 6),
                    (0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (6, 7)}

    def squares_of_assignments(self, reach_or_avoid_set):
        squares_list = []

        for active_props in list(reach_or_avoid_set):
            intersection = set.intersection(*[self.ATTACKED_SQUARES[prop] for prop in active_props])
            others = set.union(*[self.ATTACKED_SQUARES[prop] for prop in self.PIECES.keys()
                                 if prop not in active_props])

            squares_list.append(intersection - others)

        return set.union(*squares_list)


assignment_vocab = {0: 'PAD', 1: 'EPSILON', 2: 'NULL', 3: 'queen', 4: 'rook', 5: 'knight', 6: 'bishop', 7: 'pawn',
                    8: 'queen&rook', 9: 'queen&bishop', 10: 'queen&pawn&bishop', 11: 'queen&rook&pawn',
                    12: 'knight&rook', 13: 'rook&bishop', 14: 'knight&bishop', 15: 'blank'}


var_names = ['bishop', 'knight', 'pawn', 'queen', 'rook']

all_possible_reach = {var: set([tuple(assignment.split("&")) for i, assignment in assignment_vocab.items()
                             if var in assignment.split("&")]) for var in var_names}

# possible_assigments = [tuple(assignment_vocab[i].split("&")) for i in range(3, 14)]

# all_possible_reach = [i for i in possible_assigments.values()]
all_possible_avoid = {var_names[i] + "|" + var_names[j]: all_possible_reach[var_names[i]] | all_possible_reach[var_names[j]] for i in range(len(var_names))
                      for j in range(i+1, len(var_names))}

print(all_possible_reach)
print(all_possible_avoid)

env = ChessEnv()
# env = ChessWorld()

# sample_range = [tuple([i]) for i in range(3, 15)]
#
# for i in range(3, 15):
#     for j in range(i+1, 15):
#         sample_range.append((i, j))


def unpack_reach_avoid_sequence(reach_avoid_sequence):
    pass


def get_avoid_squares(avoid):
    return {("squares", "to avoid")}


def get_reach_squares(reach):
    return {(0, 1)}


def check_tasks_satisfiability(chess_env, reach_avoid_sequence):
    starting_squares = [(0, 7)]
    possible_moves = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (i != 0 or j != 0)]

    def filter_squares(row, col):
        return 0 <= row < 8 and 0 <= col < 8

    for reach_avoid in reach_avoid_sequence:
        reach, avoid = reach_avoid
        # print(reach)
        # print(avoid)
        squares_to_avoid = chess_env.squares_of_assignments(reach)
        squares_to_reach = chess_env.squares_of_assignments(avoid)
        visited = set(starting_squares) | squares_to_avoid
        queue = [i for i in starting_squares]

        starting_squares = []

        while queue and len(visited) < 64:
            cur_square = queue.pop(0)
            cur_row, cur_col = cur_square

            visited.add(cur_square)

            if cur_square in squares_to_reach:
                starting_squares.append(cur_square)

            else:

                for move_row, move_col in possible_moves:
                    new_row, new_col = cur_row + move_row, cur_col + move_col
                    new_square = (new_row, new_col)

                    if filter_squares(new_row, new_col) and new_square not in visited:
                        queue.append(new_square)

    if len(starting_squares) == 0:
        return False

    return True


all = 0
counter = 0

for name_r1, reach_1 in all_possible_reach.items():
    for name_a1, avoid_1 in {name: i for name, i in all_possible_avoid.items() if not reach_1.issubset(i)}.items():
        for name_r2, reach_2 in {name: i for name, i in all_possible_reach.items() if not (reach_1.issubset(i) and i.issubset(reach_1))}.items():
            for name_a2, avoid_2 in {name: i for name, i in all_possible_avoid.items() if not (reach_1.issubset(i) or reach_2.issubset(i))}.items():
                ra_seq = [(reach_1, avoid_1), (reach_2, avoid_2)]
                ra_name = [(name_r1, name_a1), (name_r2, name_a2)]

                all += 1
                if not check_tasks_satisfiability(env, ra_seq):
                    counter += 1
                    print(ra_name)

print(counter)
print(all)

# print(env.)
