from sympy.logic.boolalg import Or, And, Not, simplify_logic
from sympy import symbols
from itertools import combinations


class ContextMaker:
    def __init__(self, assignment_vocab, var_names, true_vars, augment_neg=()):
        self.contexts = {}
        self.assignment_vocab = assignment_vocab
        self.var_names = var_names
        self.sympy_vars = {name: symbols(name) for name in self.var_names}
        # print(self.sympy_vars)

        try:
            self.true_vars = true_vars + list(augment_neg)
        except TypeError:
            self.true_vars = true_vars | set(augment_neg)

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

            if '!' in var:
                actual_var = var.split('!')[1]
                self.sympy_vars[var] = Not(*[self.sympy_vars[actual_var]])

                for assignment, assignment_name in self.assignment_vocab.items():
                    if (actual_var not in assignment_name.split("&")) and (assignment_name not in ['PAD', 'EPSILON', 'NULL']):
                        cur_assignment_set.append(assignment)
            else:
                for assignment, assignment_name in self.assignment_vocab.items():
                    if var in assignment_name.split("&"):
                        cur_assignment_set.append(assignment)

            self.complete_var_assignments[var] = set(cur_assignment_set)

        print(self.sympy_vars)

        self.complete_assignment = set.union(*list(self.complete_var_assignments.values()))

        self.cache = {}
        self.formula_kinds = {}
        self.blank_formula_kinds = {}

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
        # print(assignments_tup)
        # print(formula)
        # print(clauses)

        # Minimize the formula
        minimized_formula = simplify_logic(formula, form="dnf")

        return minimized_formula

    def add_to_kinds_cache(self, assignment_tup, key, name, positive):
        if key not in self.formula_kinds[name][positive]:
            self.formula_kinds[name][positive][key] = []

        self.formula_kinds[name][positive][key].append(assignment_tup)

    def or_all_vars(self):
        # not_breaking_point = float(len(self.true_vars)) / 2
        tot_added = 0
        name = "or"
        self.formula_kinds[name] = {"positive": {}, "negative": {}}

        for num_vars in range(1, len(self.true_vars) - 1):
            # if num_vars <= not_breaking_point:
            tuples = list(combinations(self.true_vars, num_vars))

            for tup in tuples:
                all_assignment_set = set.union(*[self.complete_var_assignments[var] for var in tup])
                all_assignment_tup = tuple(sorted(list(all_assignment_set)))

                if all_assignment_tup not in self.cache:
                    self.cache[all_assignment_tup] = Or(*[self.sympy_vars[var] for var in tup])
                    tot_added += 1

                    self.add_to_kinds_cache(all_assignment_tup, num_vars, name, "positive")

                neg_assignment_set = self.complete_assignment - all_assignment_set
                neg_assignment_tup = tuple(sorted(list(neg_assignment_set)))

                # neg_vars = list(set(self.true_vars) - set(tup))

                if neg_assignment_tup not in self.cache:
                    self.cache[neg_assignment_tup] = Not(Or(*[self.sympy_vars[var] for var in tup]))
                    tot_added += 1

                    self.add_to_kinds_cache(neg_assignment_tup, num_vars, name, "negative")

                    # assert(neg_assignment_set | all_assignment_set == self.complete_assignment)

        # print("Or", tot_added)

    def and_all_vars(self):
        max_intersection = max([len(x.split("&")) for x in self.assignment_vocab.values()])
        tot_added = 0

        name = "and"
        self.formula_kinds[name] = {"positive": {}, "negative": {}}

        for num_vars in range(2, max_intersection + 1):
            tuples = list(combinations(self.true_vars, num_vars))

            for tup in tuples:
                all_assignment_set = set.intersection(*[self.complete_var_assignments[var] for var in tup])
                neg_assignment_set = self.complete_assignment - all_assignment_set

                if len(all_assignment_set) > 0:
                    all_assignment_tup = tuple(sorted(list(all_assignment_set)))
                    neg_assignment_tup = tuple(sorted(list(neg_assignment_set)))

                    if all_assignment_tup not in self.cache:
                        self.cache[all_assignment_tup] = And(*[self.sympy_vars[var] for var in tup])
                        tot_added += 1

                        self.add_to_kinds_cache(all_assignment_tup, num_vars, name, "positive")

                    if neg_assignment_tup not in self.cache:
                        self.cache[neg_assignment_tup] = Not(And(*[self.sympy_vars[var] for var in tup]))
                        tot_added += 1

                        self.add_to_kinds_cache(neg_assignment_tup, num_vars, name, "negative")

        # print("And", tot_added)

    def and_y_or_x(self, x=4, y=2):
        tot_added = 0

        name = "or_x_and_y"
        self.formula_kinds[name] = {"positive": {}, "negative": {}}

        for r1 in range(2, x + 1):
            for r2 in range(1, y + 1):
                or_tuples = list(combinations(self.true_vars, r1))

                for or_tup in or_tuples:
                    or_formula = Or(*[self.sympy_vars[var] for var in or_tup])
                    or_assignment_set = set.union(*[self.complete_var_assignments[var] for var in or_tup])

                    remaining_vars = list(set(self.true_vars) - set(or_tup))
                    and_tuples = list(combinations(remaining_vars, r2))

                    for neg_tup in and_tuples:
                        and_formula = And(*[self.sympy_vars[var] for var in neg_tup])
                        and_assignment_set = set.intersection(*[self.complete_var_assignments[var] for var in neg_tup])

                        final_assignment_set = or_assignment_set & and_assignment_set
                        final_assignment_tup = tuple(sorted(list(final_assignment_set)))

                        final_formula = And(*[or_formula, and_formula])

                        if len(final_assignment_set) > 0 and final_assignment_tup not in self.cache:

                            self.cache[final_assignment_tup] = final_formula
                            tot_added += 1

                            self.add_to_kinds_cache(final_assignment_tup, (r1, r2), name, "positive")

                        opposite_assignment_set = self.complete_assignment - final_assignment_set
                        opposite_assignment_tup = tuple(sorted(list(opposite_assignment_set)))

                        if len(opposite_assignment_set) < len(self.complete_assignment) and opposite_assignment_tup not in self.cache:

                            self.cache[opposite_assignment_tup] = Not(final_formula)
                            tot_added += 1

                            self.add_to_kinds_cache(opposite_assignment_tup, (r1, r2), name, "negative")

    def or_x_and_not_y(self, x=3, y=3):
        tot_added = 0
        name = "or_x_and_not_y"
        self.formula_kinds[name] = {"positive": {}, "negative": {}}

        for r1 in range(1, x+1):
            for r2 in range(1, y+1):
                positive_tuples = list(combinations(self.true_vars, r1))

                if (r1 + r2) > len(self.true_vars):
                    continue

                # print(r1, r2)

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

                            self.add_to_kinds_cache(final_assignment_tup, (r1, r2), name, "positive")

                        opposite_assignment_set = self.complete_assignment - final_assignment_set
                        opposite_assignment_tup = tuple(sorted(list(opposite_assignment_set)))

                        if opposite_assignment_tup not in self.cache:
                            self.cache[opposite_assignment_tup] = Or(*[Not(positive_formula), negative_formula])
                            # self.cache [opposite_assignment_tup] = Not(final_formula)
                            tot_added += 1

                            self.add_to_kinds_cache(opposite_assignment_tup, (r1, r2), name, "negative")

        # print("X_and_not_y", tot_added)

    def and_x_and_not_y(self, x=2, y=3):
        tot_added = 0
        name = "and_x_and_not_y"
        self.formula_kinds[name] = {"positive": {}, "negative": {}}

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

                                self.add_to_kinds_cache(final_assignment_tup, (r1, r2), name, "positive")

                            opposite_assignment_set = self.complete_assignment - final_assignment_set
                            opposite_assignment_tup = tuple(sorted(list(opposite_assignment_set)))

                            if opposite_assignment_tup not in self.cache and len(opposite_assignment_set) < len(self.complete_assignment):
                                self.cache[opposite_assignment_tup] = Or(*[Not(positive_formula), negative_formula])
                                tot_added += 1

                                self.add_to_kinds_cache(opposite_assignment_tup, (r1, r2), name, "negative")

        # print("Nx_not_y", tot_added)

    def or_x_and_not_ny(self, x=3, y=3, z=2):
        tot_added = 0
        name = "or_x_and_not_ny"
        self.formula_kinds[name] = {"positive": {}, "negative": {}}

        for r1 in range(1, x + 1):
            for r2 in range(2, y + 1):
                positive_tuples = list(combinations(self.true_vars, r1))

                # if r2 > r1:
                #     continue

                for pos_tup in positive_tuples:
                    positive_formula = Or(*[self.sympy_vars[var] for var in pos_tup])
                    positive_assignment_set = set.union(*[self.complete_var_assignments[var] for var in pos_tup])

                    # cur_vars = list(pos_tup)

                    negative_tuples = list(combinations(self.true_vars, r2))
                    actual_neg_tuples = []

                    for neg_tup in negative_tuples:
                        cur_neg_set = set.intersection(
                            *[self.complete_var_assignments[var] for var in neg_tup])

                        relevant_info = set.intersection(*[positive_assignment_set, cur_neg_set])

                        if len(cur_neg_set) > 0 and len(relevant_info) > 0:
                            actual_neg_tuples.append(neg_tup)

                    for r3 in range(1, z+1):
                        tot_tups_of_ands = list(combinations(actual_neg_tuples, r3))

                        for tup_of_ands in tot_tups_of_ands:
                            clauses_and = []
                            negative_assignment_set = set([])

                            for neg_tup in tup_of_ands:
                                cur_negative_formula = And(*[self.sympy_vars[var] for var in neg_tup])
                                cur_neg_assignment_set = set.intersection(
                                    *[self.complete_var_assignments[var] for var in neg_tup])

                                clauses_and.append(cur_negative_formula)

                                negative_assignment_set = negative_assignment_set | cur_neg_assignment_set

                            if any(all(var in neg_tup for neg_tup in tup_of_ands) for var in self.true_vars):
                                negative_formula = simplify_logic(Or(*clauses_and))
                            else:
                                negative_formula = simplify_logic(Or(*clauses_and), form='dnf')

                            if len(negative_assignment_set) > 0:

                                final_assignment_set = positive_assignment_set - negative_assignment_set
                                final_assignment_tup = tuple(sorted(list(final_assignment_set)))
                                final_formula = And(*[positive_formula, Not(negative_formula)])

                                if final_assignment_tup not in self.cache and len(final_assignment_set) > 0:
                                    self.cache[final_assignment_tup] = final_formula
                                    tot_added += 1

                                    self.add_to_kinds_cache(final_assignment_tup, (r1, r2, r3), name, "positive")

                                opposite_assignment_set = self.complete_assignment - final_assignment_set
                                opposite_assignment_tup = tuple(sorted(list(opposite_assignment_set)))

                                if opposite_assignment_tup not in self.cache and len(opposite_assignment_set) < len(
                                        self.complete_assignment):
                                    self.cache[opposite_assignment_tup] = Or(*[Not(positive_formula), negative_formula])
                                    tot_added += 1

                                    self.add_to_kinds_cache(opposite_assignment_tup, (r1, r2, r3), name, "negative")

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
            new_assignments = list(set(list(assignments) + [blank_index]))
            new_assignments = tuple(sorted(new_assignments))

            blank = self.sympy_vars['blank']

            if new_assignments not in self.cache:
                d[new_assignments] = Or(*[formula, blank])

        for assignments, formula in d.items():
            if assignments not in self.cache:
                self.cache[assignments] = formula

        kinds = {}

        for name, d1 in self.formula_kinds.items():
            cur_name_d = {}

            for pos, pos_d in d1.items():
                cur_pos_d = {}

                for shape, shape_list in pos_d.items():
                    cur_shape_list = []

                    for assignments in shape_list:
                        new_assignments = list(assignments) + [blank_index]
                        new_assignments = tuple(sorted(new_assignments))

                        cur_shape_list.append(new_assignments)

                    cur_pos_d[shape] = cur_shape_list

                cur_name_d[pos] = cur_pos_d

            kinds[name] = cur_name_d

        self.blank_formula_kinds = kinds


    def generate_cache(self):
        self.cache[tuple([])] = Or(*[])

        self.or_all_vars()
        self.and_all_vars()
        self.and_y_or_x()
        self.and_x_and_not_y()
        self.or_x_and_not_y()
        self.or_x_and_not_ny()
        # self.complete_pairs()
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

    def assignment_from_formula(self, formula):

        def inorder(node):
            if node.is_Atom:
                try:
                    return self.complete_var_assignments[str(node)]
                except KeyError:
                    if str(node) == "blank":
                        return {15}
                    else:
                        return set([])

            else:
                if node.func == Or:
                    return set.union(*[inorder(child) for child in node.args])
                elif node.func == And:
                    return set.intersection(*[inorder(child) for child in node.args])
                elif node.func == Not:
                    assert (len(node.args) == 1)
                    return self.complete_assignment - inorder(node.args[0])
                else:
                    raise ValueError("Only [And, Or, Not] are acceptable operators")

        return inorder(formula)

    def check_cache_correctness(self):
        count = 0
        for tup, formula in self.cache.items():
            tup_set = set(tup)

            true_assignment = self.assignment_from_formula(formula)

            try:
                assert (tup_set.issubset(true_assignment) and true_assignment.issubset(tup_set))
                count += 1
            except AssertionError:
                print(tup_set, true_assignment, formula)
                return False
        return count


if __name__ == "__main__":
    sample_vocab = {0: 'PAD', 1: 'EPSILON', 2: 'NULL', 3: 'queen', 4: 'rook', 5: 'knight', 6: 'bishop', 7: 'pawn',
                    8: 'queen&rook', 9: 'queen&bishop', 10: 'queen&pawn&bishop', 11: 'queen&pawn&rook',
                    12: 'knight&rook', 13: 'bishop&rook', 14: 'knight&bishop', 15: 'blank'}

    var_names = ['EPSILON', 'NULL', 'queen', 'rook', 'knight', 'bishop', 'pawn', 'blank']
    true_vars = ['queen', 'rook', 'knight', 'bishop', 'pawn']

    context_maker = ContextMaker(sample_vocab, var_names, true_vars)

    sample_voc_2 = {0: 'PAD', 1: 'EPSILON', 2: 'NULL', 3: 'red', 4: 'magenta', 5: 'magenta&red', 6: 'blue', 7: 'green',
                    8: 'aqua', 9: 'green&blue', 10: 'green&aqua', 11: 'aqua&blue', 12: 'green&aqua&blue', 13: 'yellow',
                    14: 'orange', 15: 'blank'}

    var_names_2 = ['aqua', 'blue', 'green', 'magenta', 'orange', 'red', 'yellow', 'EPSILON', 'NULL', 'blank']
    true_vars_2 = ['aqua', 'blue', 'green', 'magenta', 'orange', 'red', 'yellow']

    cm_2 = ContextMaker(sample_voc_2, var_names_2, true_vars_2)
    cm_2.generate_cache()


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

    """lens for each multiset size"""
    # for i in range(13):
    #     print(i, d[i])

    # print(len(context_maker.cache))

    # print(context_maker.formula_kinds)
    # print(context_maker.blank_formula_kinds)

    """Print some lens utilities"""
    # print()
    # cur_count = 0
    # for name, name_d in context_maker.formula_kinds.items():
    #     # other_name_d = context_maker.blank_formula_kinds[name]
    #
    #     print(name)
    #     # print(name_d)
    #     # print(other_name_d)
    #     #
    #     # print()
    #
    #     for shape, cur_list in name_d["positive"].items():
    #         if not isinstance(shape, tuple):
    #             shape = tuple([shape])
    #
    #         # if all(x in [1, 2] for x in shape):
    #         cur_count += len(cur_list)
    #         print(shape, 2 * len(cur_list), cur_count, 2 * cur_count)
    #     print()
    #
    # for assignment in context_maker.formula_kinds["or_x_and_y"]["positive"][(2, 1)]:
    #     print(assignment, context_maker.cache[assignment])

    final_count = context_maker.check_cache_correctness()
    print("Cache length:", len(context_maker.cache))
    print("Cache checked formulae:", final_count)

    for x, y in cm_2.cache.items():
        print(x, y)

    print(cm_2.make_formula((6, 7, 9, 10, 11)))
