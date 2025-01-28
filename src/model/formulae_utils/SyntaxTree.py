from utils import memory
import torch
import torch.nn.functional as F
import torch_geometric
from sympy import symbols
from sympy.logic.boolalg import Or, And, Not, simplify_logic
from model.formulae_utils.DisjointSetUnion import DisjointSetUnion, partition_colors
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# TODO: only consider ACTIVE VARIABLES when making formula out of assignments


class SyntaxTree:
    def __init__(self, variable_names, assignment_vocab):
        self.variable_names = variable_names
        self.sympy_vars = {name: symbols(name) for name in self.variable_names}
        self.assignment_vocab = assignment_vocab
        self.embedding_dict = {i+1: k for i, k in enumerate(variable_names)}
        self.contexts, self.dsu = partition_colors(list(self.assignment_vocab.values()))

        for operator in ["AND", "OR", "NOT"]:
            self.embedding_dict[len(self.embedding_dict)+1] = operator

        # Will use the built-ind padding-idx of embedding layer to encode FALSE as an avoid sequence (no constraints)
        self.embedding_dict[0] = "FALSE"

        self.in_mem_cache = {(i, j): self.build_syntax_tree((i, j)) for i in range(3, len(assignment_vocab) - 1)
                             for j in range(3, len(assignment_vocab) - 1)}

        self.in_mem_cache[tuple([])] = self.build_syntax_tree(tuple([]))

        for i in range(3, len(assignment_vocab) - 1):
            self.in_mem_cache[(i, )] = self.build_syntax_tree((i, ))

        self.vars_assignment_sets = {var: (set([i for i, assignment in self.assignment_vocab.items()
                                                        if var in assignment.split("&")]))
                                     for var in self.variable_names}

        for i in range(len(self.variable_names)):
            for j in range(i+1, len(self.variable_names)):
                var1 = self.variable_names[i]
                var2 = self.variable_names[j]

                cur_set = self.vars_assignment_sets[var1] | self.vars_assignment_sets[var2]
                cur_assignment_set = tuple(sorted(list(cur_set)))

                self.in_mem_cache[cur_assignment_set] = self.build_syntax_tree(cur_assignment_set)

        for var, cur_set in self.vars_assignment_sets.items():
            cur_assignment_set = tuple(sorted(list(cur_set)))
            self.in_mem_cache[cur_assignment_set] = self.build_syntax_tree(cur_assignment_set)

        for var, cur_set in self.vars_assignment_sets.items():
            for i in range(3, len(assignment_vocab) - 1):
                cur_set_complete = cur_set | {i}
                cur_assignment_set = tuple(sorted(list(cur_set_complete)))

                if cur_assignment_set not in self.in_mem_cache:
                    self.in_mem_cache[cur_assignment_set] = self.build_syntax_tree(cur_assignment_set)



    @staticmethod
    def simplify_formula(formula):
        return simplify_logic(formula, form='dnf')

    def read_proposition(self, var):
        assignment_name = self.assignment_vocab[var]

        if "&" in assignment_name:
            and_vars = assignment_name.split("&")
            return And(*[self.sympy_vars[var_name] for var_name in and_vars])

        return self.sympy_vars[assignment_name]

    def process_proposition_seq(self, assignment_set_seq):
        graphs = []

        for assignment_set in assignment_set_seq:
            tup_assignment = tuple(sorted([i.item() for i in assignment_set
                                           if i.item() not in [0, 1, 2,
                                           len(self.assignment_vocab) - 1,
                                           ]]))
            # formula = Or(*[
            #     self.read_proposition(var.item())
            #     for var in assignment_set if var.item() not in [0, 1, 2, len(self.assignment_vocab)-1]
            # ])

            graphs.append(self.build_syntax_tree(tup_assignment))

        return graphs

    def build_syntax_tree(self, assignment_set):

        # @memory.cache
        def cheat_syntax_tree(assignment_set):
            formula = Or(*[
                self.read_proposition(var)
                for var in assignment_set
            ])

            formula = self.simplify_formula(formula)

            X, edge_index = self.syntax_tree_from_formula(formula)

            return X, edge_index

        X, edge_index = cheat_syntax_tree(assignment_set)

        # return X, edge_index
        return Data(x=torch.tensor(X, dtype=torch.long), edge_index=edge_index)

    def syntax_tree_from_formula(self, formula):
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

        return X, edge_index

    def process_sequence(self, seqs, lens):
        """Idea on how to speed up:
           flatten seqs, map function to each reach-set
           use torch batching and generate roots from batch.batch
        """
        # node_features = []
        # edge_index = torch.tensor([[], []], dtype=torch.long)
        # roots = []
        # batching_factor = 0

        # max_len = max(lens)
        #
        # for seq, length in zip(seqs, lens):
        #     actual_seq = seq[:length]
        #
        #     graphs = self.process_proposition_seq(actual_seq)
        #
        #     for X, edges in graphs:
        #         roots.append(batching_factor)
        #         edges += batching_factor
        #
        #         batching_factor += len(X)
        #
        #         node_features += X
        #         edge_index = torch.cat([edge_index, edges], dim=1)
        #
        #     for _ in range(max_len - len(graphs)):
        #         roots.append(0)

        # return torch.tensor(node_features, dtype=torch.long), edge_index, roots

        all_seqs = seqs.flatten(start_dim=0, end_dim=1)
        tot_len = all_seqs.shape[0]

        def build_graph(reach_set):
            reach_tup = tuple(sorted([i.item() for i in reach_set
                                      if i.item() not in
                                      [0, 1, 2,
                                       len(self.assignment_vocab) - 1,
                                       ]]))

            if len(reach_tup) < 3:
                return self.in_mem_cache[reach_tup]

            return self.build_syntax_tree(reach_tup)

        # build_graph = lambda reach_set: self.in_mem_cache[tuple(sorted([i.item() for i in reach_set
        #                                                                   if i.item() not in
        #                                                                   [0, 1, 2,
        #                                                                    len(self.assignment_vocab) - 1,
        #                                                                    ]]))]
        all_graphs = list(map(build_graph, all_seqs))
        dl = DataLoader(all_graphs, batch_size=tot_len+2, shuffle=False)

        for btch in dl:
            batch = btch.batch
            X = btch.x
            edge_index = btch.edge_index

        almost_roots = torch.cumsum(torch.bincount(batch), dim=0)
        roots = torch.cat((torch.zeros(1, dtype=torch.long), almost_roots), dim=0)[:-1]

        return X, edge_index, roots
