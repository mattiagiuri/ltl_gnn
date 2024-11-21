# import torch
# from preprocessing import BatchedReachAvoidSequences
from sympy import symbols
from sympy.logic.boolalg import Or, And, Not, simplify_logic

class SyntaxTree:

    # def __init__(self, batched_seqs: tuple[tuple[torch.tensor, torch.tensor], tuple[torch.tensor, torch.tensor]]
    #                                 | BatchedReachAvoidSequences = None):
    #     if isinstance(batched_seqs, BatchedReachAvoidSequences):
    #         (reach_lens, reach_data), (avoid_lens, avoid_data) = batched_seqs.all()
    #     else:
    #         (reach_lens, reach_data), (avoid_lens, avoid_data) = batched_seqs
    #     assert (reach_lens == avoid_lens).all()

    def __init__(self):
        pass


    @staticmethod
    def minimal_formula(assignments):
        variable_names = ["x" + str(i) for i in range(len(assignments[0]))]
        sympy_vars = {name: symbols(name) for name in variable_names}

        clauses = []
        for assignment in assignments:
            # Create a conjunctive clause for the assignment
            clause = And(*[
                sympy_vars[var] if value == 1 else Not(sympy_vars[var])
                for var, value in zip(variable_names, assignment)
            ])
            clauses.append(clause)

        # Combine clauses into a disjunction
        formula = Or(*clauses)

        # Minimize the formula
        minimized_formula = simplify_logic(formula, form='dnf')

        return minimized_formula, variable_names

    def syntax_tree(self, assignments):
        formula, variable_names = self.minimal_formula(assignments)

        n = len(variable_names)
        sympy_vars = {name: i for i, name in enumerate(variable_names)}
        one_hot_length = n + 3  # Variables + AND, OR, NOT

        # Outputs
        X = []  # Node features
        edge_index = [[], []]  # Edges (directed)

        # Helper function for inorder traversal
        def inorder(node):
            nonlocal one_hot_length
            nonlocal sympy_vars
            nonlocal X, edge_index

            if node.is_Atom:  # Leaf node (variable)
                var_index = sympy_vars[str(node)]  # Get the index of the variable
                one_hot = [0] * one_hot_length
                one_hot[var_index] = 1
                current_index = len(X)
                X.append(one_hot)  # Add node feature

            else:  # Operator node
                if node.func == Or:
                    op_index = n + 1
                elif node.func == Not:
                    op_index = n + 2
                elif node.func == And:
                    op_index = n
                else:
                    raise ValueError("Only [And, Or, Not] are acceptable operators")

                one_hot = [0] * one_hot_length
                one_hot[op_index] = 1
                current_index = len(X)
                X.append(one_hot)  # Add node feature

                # Add edges from children
                for arg in node.args:
                    child_index = inorder(arg)
                    edge_index[0].append(child_index)  # From child (arg node)
                    edge_index[1].append(current_index)  # To parent (current node)

            # Return the index of the current node in X
            return current_index

        # Perform inorder traversal starting from the root
        inorder(formula)

        return X, edge_index

tree = SyntaxTree()
assignments = [(1, 0, 0, 1), (0, 1, 0, 1), (1, 1, 0, 1), (0, 0, 0, 1)]

print(tree.minimal_formula(assignments))
print(tree.syntax_tree(assignments))
