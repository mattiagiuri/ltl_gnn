from sympy.logic.boolalg import Or, And, Not, simplify_logic
from sympy import symbols
from itertools import combinations


class ContextMaker:
    def __init__(self, assignment_vocab, var_names, true_vars):
        self.contexts = {}
        self.assignment_vocab = assignment_vocab
        self.var_names = var_names
        self.sympy_vars = {name: symbols(name) for name in self.var_names}
        self.true_vars = true_vars

        self.ids = {assignment: set(assignment.split("&")) for assignment in assignment_vocab.values()}

        for cur, assignment in assignment_vocab.items():
            id = self.ids[assignment]
            cur_context = set([])

            for tmp, tmp_id in self.ids.items():
                if id.issubset(tmp_id):
                    set_diff = tmp_id - id
                    cur_context = cur_context | set_diff

            self.contexts[cur] = cur_context

        self.complete_var_assignments = {}

        for var in self.true_vars:
            cur_assignment_set = []
            for assignment, assignment_name in self.assignment_vocab.items():
                if var in assignment_name.split("&"):
                    cur_assignment_set.append(assignment)

            self.complete_var_assignments[var] = set(cur_assignment_set)

        self.complete_assignment = set.union(*list(self.complete_var_assignments.values()))

        self.cache = {}

    def make_formula(self, assignments_tup):
        clauses = []
        for assignment in assignments_tup:
            context = self.contexts[assignment]
            assignment_set = set(self.assignment_vocab[assignment].split("&"))

            positives = [self.sympy_vars[var] for var in assignment_set]
            negatives = [Not(self.sympy_vars[var]) for var in context]

            all = positives + negatives

            # print(all)

            clause = And(*all)
            clauses.append(clause)

        # Combine clauses into a disjunction
        formula = Or(*clauses)

        # Minimize the formula
        minimized_formula = simplify_logic(formula, form="dnf")

        return minimized_formula

    def or_all_vars(self):
        # not_breaking_point = float(len(self.true_vars)) / 2
        tot_added = 0
        for num_vars in range(1, len(self.true_vars) - 1):
            # if num_vars <= not_breaking_point:
            tuples = list(combinations(self.true_vars, num_vars))

            for tup in tuples:
                all_assignment_set = set.union(*[self.complete_var_assignments[var] for var in tup])
                all_assignment_tup = tuple(sorted(list(all_assignment_set)))

                if all_assignment_tup not in self.cache:
                    self.cache[all_assignment_tup] = Or(*[self.sympy_vars[var] for var in tup])
                    tot_added += 1

                neg_assignment_set = self.complete_assignment - all_assignment_set
                neg_assignment_tup = tuple(sorted(list(neg_assignment_set)))

                # neg_vars = list(set(self.true_vars) - set(tup))

                if neg_assignment_tup not in self.cache:
                    self.cache[neg_assignment_tup] = Not(Or(*[self.sympy_vars[var] for var in tup]))
                    tot_added += 1

                    # assert(neg_assignment_set | all_assignment_set == self.complete_assignment)

        # print("Or", tot_added)

    def and_all_vars(self):
        max_intersection = max([len(x.split("&")) for x in self.assignment_vocab.values()])
        tot_added = 0

        for i in range(2, max_intersection + 1):
            tuples = list(combinations(self.true_vars, i))

            for tup in tuples:
                all_assignment_set = set.intersection(*[self.complete_var_assignments[var] for var in tup])
                neg_assignment_set = self.complete_assignment - all_assignment_set

                if len(all_assignment_set) > 0:
                    all_assignment_tup = tuple(sorted(list(all_assignment_set)))
                    neg_assignment_tup = tuple(sorted(list(neg_assignment_set)))

                    if all_assignment_tup not in self.cache:
                        self.cache[all_assignment_tup] = And(*[self.sympy_vars[var] for var in tup])
                        tot_added += 1

                    if neg_assignment_tup not in self.cache:
                        self.cache[neg_assignment_tup] = Not(And(*[self.sympy_vars[var] for var in tup]))
                        tot_added += 1

        # print("And", tot_added)

    def or_x_and_not_y(self, x=2, y=3):
        tot_added = 0
        for r1 in range(1, x+1):
            for r2 in range(1, y+1):
                positive_tuples = list(combinations(self.true_vars, r1))

                if (r1 + r2) > len(self.true_vars):
                    continue

                for pos_tup in positive_tuples:
                    positive_formula = Or(*[self.sympy_vars[var] for var in pos_tup])
                    positive_assignment_set = set.union(*[self.complete_var_assignments[var] for var in pos_tup])

                    remaining_vars = list(set(self.true_vars) - set(pos_tup))
                    negative_tuples = list(combinations(remaining_vars, r2))

                    for neg_tup in negative_tuples:
                        negative_formula = Or(*[self.sympy_vars[var] for var in neg_tup])
                        negative_assignment_set = set.union(*[self.complete_var_assignments[var] for var in neg_tup])

                        final_assignment_set = positive_assignment_set - negative_assignment_set
                        final_assignment_tup = tuple(sorted(list(final_assignment_set)))
                        final_formula = And(*[positive_formula, Not(negative_formula)])

                        if final_assignment_tup not in self.cache:
                            self.cache[final_assignment_tup] = final_formula
                            tot_added += 1

                        opposite_assignment_set = self.complete_assignment - final_assignment_set
                        opposite_assignment_tup = tuple(sorted(list(opposite_assignment_set)))

                        if opposite_assignment_tup not in self.cache:
                            self.cache[opposite_assignment_tup] = Or(*[Not(positive_formula), negative_formula])
                            tot_added += 1

        # print("X_and_not_y", tot_added)

    def or_nx_and_not_y(self, x=2, y=3):
        tot_added = 0
        for r1 in range(2, x + 1):
            for r2 in range(1, y + 1):
                positive_tuples = list(combinations(self.true_vars, r1))

                if (r1 + r2) > len(self.true_vars):
                    continue

                for pos_tup in positive_tuples:
                    positive_formula = And(*[self.sympy_vars[var] for var in pos_tup])
                    positive_assignment_set = set.intersection(*[self.complete_var_assignments[var] for var in pos_tup])

                    if len(positive_assignment_set) > 0:

                        remaining_vars = list(set(self.true_vars) - set(pos_tup))
                        negative_tuples = list(combinations(remaining_vars, r2))

                        for neg_tup in negative_tuples:
                            negative_formula = Or(*[self.sympy_vars[var] for var in neg_tup])
                            negative_assignment_set = set.union(*[self.complete_var_assignments[var] for var in neg_tup])

                            final_assignment_set = positive_assignment_set - negative_assignment_set
                            final_assignment_tup = tuple(sorted(list(final_assignment_set)))
                            final_formula = And(*[positive_formula, Not(negative_formula)])

                            if final_assignment_tup not in self.cache and len(final_assignment_set) > 0:
                                self.cache[final_assignment_tup] = final_formula
                                tot_added += 1

                            opposite_assignment_set = self.complete_assignment - final_assignment_set
                            opposite_assignment_tup = tuple(sorted(list(opposite_assignment_set)))

                            if opposite_assignment_tup not in self.cache and len(opposite_assignment_set) < len(self.complete_assignment):
                                self.cache[opposite_assignment_tup] = Or(*[Not(positive_formula), negative_formula])
                                tot_added += 1

        # print("Nx_not_y", tot_added)

    def complete_pairs(self):
        tot_added = 0

        for i in range(3, len(self.assignment_vocab) - 1):
            for j in range(i + 1, len(self.assignment_vocab) - 1):
                formula_list = [self.cache[(i,)], self.cache[(j,)]]

                if (i, j) not in self.cache:
                    self.cache[(i, j)] = simplify_logic(Or(*formula_list), form="dnf")
                    tot_added += 1

                neg_assignment_set = self.complete_assignment - {i, j}
                neg_assignment_tup = tuple(sorted(list(neg_assignment_set)))

                if neg_assignment_tup not in self.cache:
                    self.cache[neg_assignment_tup] = simplify_logic(Not(Or(*formula_list)))
                    tot_added += 1

        for var in self.true_vars:
            for i in range(3, len(self.assignment_vocab) - 1):
                var_set = self.complete_var_assignments[var]

                if i not in var_set:
                    cur_assignments = var_set | {i}
                    cur_assignments_tup = tuple(sorted(list(cur_assignments)))

                    formula_list = [self.cache[tuple(sorted(list(var_set)))], self.cache[(i, )]]

                    if cur_assignments_tup not in self.cache:
                        self.cache[cur_assignments_tup] = Or(*formula_list)
                        tot_added += 1

                    neg_assignment_set = self.complete_assignment - cur_assignments
                    neg_assignment_tup = tuple(sorted(list(neg_assignment_set)))

                    if neg_assignment_tup not in self.cache:
                        self.cache[neg_assignment_tup] = simplify_logic(Not(Or(*formula_list)))
                        tot_added += 1

        # print(tot_added)

    def add_blanks(self):
        blank_index = len(self.assignment_vocab) - 1
        assert(self.assignment_vocab[blank_index] == 'blank')

        d = {}

        for assignments, formula in self.cache.items():
            new_assignments = list(assignments) + [blank_index]
            new_assignments = tuple(sorted(new_assignments))

            blank = self.sympy_vars['blank']
            d[new_assignments] = Or(*[formula, blank])

        for assignments, formula in d.items():
            self.cache[assignments] = formula

    def generate_cache(self):
        self.cache[tuple([])] = Or(*[])

        self.or_all_vars()
        self.and_all_vars()
        self.or_nx_and_not_y()
        self.or_x_and_not_y()
        self.complete_pairs()
        self.add_blanks()

        return self.cache

    def generate_piece_set(self, assignments_tup):
        if len(assignments_tup) == 0:
            return {}

        assignments_names = [self.assignment_vocab[i] for i in assignments_tup]

        pieces_dict = {var: set([x for x in assignments_names if var in x]) for var in self.var_names}
        pieces_dict = {var: active for var, active in pieces_dict.items() if len(active) > 0}

        active_pieces = list(pieces_dict.keys())
        num_active_pieces = len(pieces_dict.keys())

        final_pieces_dict = {}
        exclude = set([])

        for i in range(num_active_pieces):
            for j in range(i+1, num_active_pieces):
                var1 = active_pieces[i]
                var2 = active_pieces[j]

                if pieces_dict[var1].issubset(pieces_dict[var2]) and pieces_dict[var2].issubset(pieces_dict[var1]):
                    final_pieces_dict[(var1, var2)] = pieces_dict[var1]
                    exclude.add(var1)
                    exclude.add(var2)

                elif pieces_dict[var1].issubset(pieces_dict[var2]):

                    final_pieces_dict[(var2, )] = pieces_dict[var2]
                    exclude.add(var1)

                elif pieces_dict[var2].issubset(pieces_dict[var1]):
                    final_pieces_dict[(var1, )] = pieces_dict[var1]
                    exclude.add(var2)

                else:
                    pass

    def assignment_wise_interactions(self, k: int):
        at_least_k_interactions = set.union(*[set(combinations(list(self.complete_var_assignments[var]), k))
                                              for var in self.true_vars])

        return at_least_k_interactions


if __name__ == "__main__":
    sample_vocab = {0: 'PAD', 1: 'EPSILON', 2: 'NULL', 3: 'queen', 4: 'rook', 5: 'knight', 6: 'bishop', 7: 'pawn',
                    8: 'queen&rook', 9: 'queen&bishop', 10: 'queen&pawn&bishop', 11: 'queen&pawn&rook',
                    12: 'knight&rook', 13: 'bishop&rook', 14: 'knight&bishop', 15: 'blank'}

    var_names = ['EPSILON', 'NULL', 'queen', 'rook', 'knight', 'bishop', 'pawn', 'blank']
    true_vars = ['queen', 'rook', 'knight', 'bishop', 'pawn']

    context_maker = ContextMaker(sample_vocab, var_names, true_vars)

    # for i, context in context_maker.contexts.items():
    #     print(sample_vocab[i] + ": " + str(context))

    # print(context_maker.make_formula((3, 8, 9, 10, 11)))
    # print(context_maker.complete_var_assignments)

    # context_maker.or_all_vars()
    # context_maker.and_all_vars()
    # context_maker.or_nx_and_not_y()
    # context_maker.or_x_and_not_y()
    # context_maker.complete_pairs()
    # context_maker.add_blanks()

    context_maker.generate_cache()

    # for assignments, formula in context_maker.cache.items():
    #     print(assignments, formula)

    # count_1 = 0
    # count_2 = 0
    #
    # for assignments, _ in context_maker.cache.items():
    #     if len(assignments) == 1:
    #         count_1 += 1
    #     elif len(assignments) == 2:
    #         count_2 += 1
    #
    #
    #
    # objective_1 = 12
    #
    # o2 = set.union(*[set(combinations(list(context_maker.complete_var_assignments[var]), 2)) for var in context_maker.true_vars])
    # objective_2 = len(o2)

    # print(1, count_1, objective_1)
    # print(2, count_2, objective_2)

    # ob2 = context_maker.assignment_wise_interactions(2)
    # cur2 = set([i for i in context_maker.cache.keys() if len(i) == 2])

    # print(ob2 - cur2)
    #
    # for assignments, formula in context_maker.cache.items():
    #     if len(assignments) == 1:
    #         print(assignments, formula)
    #
    # for assignments, formula in context_maker.cache.items():
    #     if len(assignments) == 2:
    #         print(assignments, formula)

    # print(simplify_logic(Or(*[context_maker.cache[(4,)], context_maker.cache[(6,)]])))

    d = {}

    for assignment in context_maker.cache.keys():
        d[len(assignment)] = d.get(len(assignment), 0) + 1

    for i in range(13):
        print(i, d[i])

    print(len(context_maker.cache))