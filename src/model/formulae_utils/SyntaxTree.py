from utils import memory
import torch
import torch.nn.functional as F
import torch_geometric
from sympy import symbols
from sympy.logic.boolalg import Or, And, Not, simplify_logic

# TODO: only consider ACTIVE VARIABLES when making formula out of assignments


class SyntaxTree:
    def __init__(self, variable_names, assignment_vocab):
        self.variable_names = variable_names
        self.sympy_vars = {name: symbols(name) for name in self.variable_names}
        self.assignment_vocab = assignment_vocab
        self.embedding_dict = {i+1: k for i, k in enumerate(variable_names)}

        for operator in ["AND", "OR", "NOT"]:
            self.embedding_dict[len(self.embedding_dict)+1] = operator

        # Will use the built-ind padding-idx of embedding layer to encode FALSE as an avoid sequence (no constraints)
        self.embedding_dict[0] = "FALSE"

    def minimal_formula(self, assignments, active_vars):
        clauses = []
        for assignment in assignments:
            # Create a conjunctive clause for the assignment
            # We use the active_vars to keep the formula readable
            # These are the variables that appear in the current set of assignments
            # TODO: discuss
            clause = And(*[
                self.sympy_vars[var] if value == 1 else Not(self.sympy_vars[var])
                for var, value in zip(self.variable_names, assignment)
                if var in active_vars
            ])
            clauses.append(clause)

        # Combine clauses into a disjunction
        formula = Or(*clauses)

        # Minimize the formula
        minimized_formula = self.simplify_formula(formula)

        return minimized_formula

    @staticmethod
    @memory.cache
    def simplify_formula(formula):
        return simplify_logic(formula)

    def read_assignment(self, var):
        assignment_name = self.assignment_vocab[var.item()]
        cur_assignment = [0 for i in range(len(self.variable_names))]

        if "&" in assignment_name:
            and_vars = assignment_name.split("&")

            for name in and_vars:
                cur_assignment[self.variable_names.index(name)] = 1

        else:
            cur_assignment[self.variable_names.index(assignment_name)] = 1

        return cur_assignment

    def get_active_vars(self, assignment_set):
        # TODO: discuss
        active_vars = []

        for var in assignment_set:
            if var.item() not in [0, 1, 2, len(self.assignment_vocab) - 1]:
                assignment_name = self.assignment_vocab[var.item()]

                if "&" in assignment_name:
                    active_vars += assignment_name.split("&")
                else:
                    active_vars.append(assignment_name)

        return active_vars

    def process_assignments_seq(self, assignment_set_seq):
        formulae = []

        for assignment_set in assignment_set_seq:
            assignments = [
                self.read_assignment(var)
                for var in assignment_set if var.item() not in [0, 1, 2, len(self.assignment_vocab)-1]
            ]

            active_vars = self.get_active_vars(assignment_set)

            formula = self.minimal_formula(assignments, active_vars)

            formulae.append(formula)

        return formulae

    def read_proposition(self, var):
        assignment_name = self.assignment_vocab[var]

        if "&" in assignment_name:
            and_vars = assignment_name.split("&")
            return And(*[self.sympy_vars[var_name] for var_name in and_vars])

        return self.sympy_vars[assignment_name]

    def process_proposition_seq(self, assignment_set_seq):
        formulae = []

        for assignment_set in assignment_set_seq:
            formula = Or(*[
                self.read_proposition(var.item())
                for var in assignment_set if var.item() not in [0, 1, 2, len(self.assignment_vocab)-1]
            ])

            formulae.append(simplify_logic(formula))

        return formulae

    def syntax_tree_from_formula(self, formula, batching_factor=0):
        n = len(self.variable_names)
        one_hot_length = n + 3  # Variables + AND, OR, NOT

        # Outputs
        X = []  # Node features
        edge_index = [[], []]  # Edges (directed)

        # Helper function for inorder traversal
        def inorder(node):
            nonlocal one_hot_length
            nonlocal X, edge_index

            if node.is_Atom:  # Leaf node (variable)
                try:
                    var_index = self.variable_names.index(str(node)) + 1  # Get the index of the variable
                except ValueError:
                    var_index = 0
                # one_hot = [0] * one_hot_length
                # one_hot[var_index] = 1
                current_index = len(X)
                X.append(var_index)  # Add node feature

            else:  # Operator node
                if node.func == Or:
                    op_index = n + 2
                elif node.func == Not:
                    op_index = n + 3
                elif node.func == And:
                    op_index = n + 1
                else:
                    raise ValueError("Only [And, Or, Not] are acceptable operators")

                # one_hot = [0] * one_hot_length
                # one_hot[op_index] = 1
                current_index = len(X)
                X.append(op_index)  # Add node feature

                # Add edges from children
                for arg in node.args:
                    child_index = inorder(arg)
                    edge_index[0].append(child_index)  # From child (arg node)
                    edge_index[1].append(current_index)  # To parent (current node)

            # Return the index of the current node in X
            return current_index

        # Perform inorder traversal starting from the root
        inorder(formula)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index += batching_factor

        return X, edge_index

    def process_sequence(self, seqs, lens, prop_mode=False):
        node_features = []
        edge_index = torch.tensor([[], []], dtype=torch.long)
        roots = []
        batching_factor = 0

        max_len = max(lens)

        for seq, length in zip(seqs, lens):
            actual_seq = seq[:length]

            if prop_mode:
                formulae = self.process_proposition_seq(actual_seq)
            else:
                formulae = self.process_assignments_seq(actual_seq)

            print(formulae)

            for formula in formulae:
                roots.append(batching_factor)

                X, edges = self.syntax_tree_from_formula(formula, batching_factor)
                batching_factor += len(X)

                node_features += X
                edge_index = torch.cat([edge_index, edges], dim=1)

            for _ in range(max_len - len(formulae)):
                roots.append(0)

        return node_features, edge_index, roots

    def process_reach_avoid(self, reachs, reach_lens, avoids, prop_mode=False):
        batching_factor = 0

        node_features = []
        edge_index = torch.tensor([[], []], dtype=torch.long)
        roots = []
        # print(reachs, avoids)
        for reach, r_len, avoid in zip(reachs, reach_lens, avoids):

            reach_seq = reach[:r_len]
            avoid_seq = avoid[:r_len]

            if prop_mode:
                formulae_reach = self.process_proposition_seq(reach_seq)
                formulae_avoid = self.process_proposition_seq(avoid_seq)
            else:
                formulae_reach = self.process_assignments_seq(reach_seq)
                formulae_avoid = self.process_assignments_seq(avoid_seq)

            formulae = [self.simplify_formula(And(*[r, Not(a)])) for r, a in zip(formulae_reach, formulae_avoid)]
            print(formulae)
            # print(formulae_reach)
            # print(formulae_avoid)
            for i, formula in enumerate(formulae):
                roots.append(batching_factor)

                # print(formula)
                X, edges = self.syntax_tree_from_formula(formula, batching_factor)
                batching_factor += len(X)

                node_features += X
                edge_index = torch.cat([edge_index, edges], dim=1)

            for _ in range(max(reach_lens) - len(formulae)):
                roots.append(0)

        return node_features, edge_index, roots
