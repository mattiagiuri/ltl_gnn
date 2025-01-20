from itertools import product
# from sequence.samplers.chessworld_sequence_samplers import chessworld_sample_reach_avoid, chessworld_all_reach_avoid
# from envs.chessworld.chessworld import ChessWorld


class ChessEnv():
    PIECES = {
        "queen": (4, 0),
        "rook": (4, 2),
        "knight": (1, 3),
        "bishop": (3, 4),
        "pawn": (2, 2)
    }

    ATTACKED_SQUARES = {
        "queen": {(4, 0), (4, 1), (4, 2), (3, 1), (2, 2),
                  (3, 0), (2, 0), (1, 0), (0, 0)},
        "rook": {(4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
                 (3, 2), (2, 2)},
        "knight": {(1, 3), (3, 2), (2, 1), (0, 1), (3, 4)},
        "bishop": {(0, 1), (1, 2), (2, 3), (3, 4), (4, 3)},
        "pawn": {(2, 2), (3, 1), (3, 3)}

    }

    FREE_SQUARES = {(0, 2), (1, 1), (0, 3), (0, 4), (1, 4), (2, 4)}

    def squares_of_assignments(self, reach_or_avoid_set):
        squares_list = []

        for active_props in reach_or_avoid_set:
            intersection = set.intersection(*[self.ATTACKED_SQUARES[prop] for prop in active_props])
            others = set.union(*[self.ATTACKED_SQUARES[prop] for prop in self.PIECES.keys()
                                 if prop not in active_props])

            squares_list.append(intersection - others)

        return set.union(*squares_list)

assignment_vocab = {0: 'PAD', 1: 'EPSILON', 2: 'NULL', 3: 'queen', 4: 'rook', 5: 'knight',
                    6: 'bishop', 7: 'pawn', 8: 'rook&queen', 9: 'pawn&queen', 10: 'rook&pawn&queen',
                    11: 'rook&knight', 12: 'bishop&rook', 13: 'bishop&knight', 14: 'blank'}

var_names = ['bishop', 'knight', 'pawn', 'queen', 'rook']
possible_assigments = [tuple(assignment_vocab[i].split("&")) for i in range(3, 14)]

all_possible_reach = {i for i in possible_assigments}
all_possible_avoid = {(possible_assigments[i], possible_assigments[j]) for i in range(len(possible_assigments))
                      for j in range(i+1, len(possible_assigments))}

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
    starting_squares = [(0, 4)]
    possible_moves = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (i != 0 or j != 0)]

    def filter_squares(row, col):
        return 0 <= row < 5 and 0 <= col < 5

    for reach_avoid in reach_avoid_sequence:
        reach, avoid = reach_avoid
        # print(reach)
        # print(avoid)
        squares_to_avoid = chess_env.squares_of_assignments(reach)
        squares_to_reach = chess_env.squares_of_assignments(avoid)
        visited = set(starting_squares) | squares_to_avoid
        queue = [i for i in starting_squares]

        starting_squares = []

        while queue and len(visited) < 25:
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


counter = 0

for reach_1 in all_possible_reach:
    for avoid_1 in {(i, j) for i, j in all_possible_avoid if i != reach_1 and j != reach_1}:
        for reach_2 in all_possible_reach - {reach_1}:
            for avoid_2 in {(i, j) for i, j in all_possible_avoid if i != reach_1 and i != reach_2
                                                                     and j != reach_1 and j != reach_2}:
                ra_seq = [((reach_1, ), avoid_1), ((reach_2, ), avoid_2)]

                if not check_tasks_satisfiability(env, ra_seq):
                    counter += 1
                    print(ra_seq)

print(counter)

# print(env.)
