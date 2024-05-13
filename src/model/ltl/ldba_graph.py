from typing import Optional

import torch
from torch_geometric.data import Data

from ltl.automata import LDBA, LDBATransition
from ltl.logic import Assignment, FrozenAssignment


class LDBAGraph(Data):
    CACHE: dict[tuple[str, int], tuple['LDBAGraph', 'LDBAGraph']] = {}

    def __init__(
            self,
            features: torch.tensor,
            edge_index: torch.tensor,
            labels: dict[int, str],
            root_assignments: set[FrozenAssignment],
            **kwargs
    ):
        super().__init__(x=features, edge_index=edge_index, **kwargs)
        self.labels = labels
        self.root_assignments = root_assignments

    @classmethod
    def from_ldba(cls, ldba: LDBA, current_state: int) -> tuple['LDBAGraph', 'LDBAGraph']:
        """Returns the positive and negative LDBA graphs."""
        if not ldba.complete:
            raise ValueError('The LDBA must be complete. Make sure to call '
                             '`ldba.complete_sink_state()` before constructing the graph.')
        if not ldba.state_to_scc:
            raise ValueError('The SCCs of the LDBA must be initialised. Make sure to call '
                             '`ldba.compute_sccs()` before constructing the graph.')
        assert ldba.formula is not None
        if (ldba.formula, current_state) in cls.CACHE:
            return cls.CACHE[(ldba.formula, current_state)]
        pos_graph = cls.construct_graph(ldba, current_state, positive=True)
        neg_graph = cls.construct_graph(ldba, current_state, positive=False)
        cls.CACHE[(ldba.formula, current_state)] = pos_graph, neg_graph
        return pos_graph, neg_graph

    @classmethod
    def construct_graph(cls, ldba: LDBA, current_state: int, positive: bool) -> 'LDBAGraph':
        transition_to_index = {}
        negative = not positive
        edges = set()

        def add_to_index(transition: LDBATransition):
            if transition not in transition_to_index:
                transition_to_index[transition] = len(transition_to_index) + 1  # 0 is root node

        def dfs(state: int, path: list[LDBATransition], state_to_path_index: dict[int, int],
                accepting_transition: Optional[LDBATransition]) -> set[LDBATransition]:
            state_to_path_index[state] = len(path)
            transitions = set()
            for transition in ldba.state_to_transitions[state]:
                scc = ldba.state_to_scc[transition.target]
                if negative and scc.bottom and not scc.accepting:
                    transitions.add(transition)
                    add_to_index(transition)
                else:
                    path.append(transition)
                    stays_in_scc = scc == ldba.state_to_scc[transition.source]
                    updated_accepting_transition = accepting_transition
                    if transition.accepting and stays_in_scc:
                        updated_accepting_transition = transition
                    if transition.target in state_to_path_index:  # found cycle
                        if positive and updated_accepting_transition in path[state_to_path_index[transition.target]:]:
                            # found accepting cycle
                            future_transitions = [path[state_to_path_index[transition.target]]]
                        else:
                            path.pop()
                            continue  # found non-accepting cycle
                    else:
                        future_transitions = dfs(transition.target, path, state_to_path_index,
                                                 updated_accepting_transition)
                    if future_transitions:
                        transitions.add(transition)
                        add_to_index(transition)
                        for t in future_transitions:
                            add_to_index(t)
                            edges.add((transition_to_index[t], transition_to_index[transition]))
                    path.pop()
            del state_to_path_index[state]
            return transitions

        root_transitions = dfs(current_state, [], {}, None)
        if not root_transitions:
            root_assignments = set()
        else:
            root_assignments = set.union(*(cls.get_assignments(t, ldba.possible_assignments) for t in root_transitions))
        edges |= {(transition_to_index[st], 0) for st in root_transitions}  # add edges to root node
        edges = torch.tensor([[], []], dtype=torch.long) if not edges \
            else torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        features = [[0] * (len(ldba.possible_assignments) + 2)]
        sorted_transitions = sorted(transition_to_index.keys(), key=lambda x: transition_to_index[x])
        features += [cls.get_features(t, ldba.possible_assignments) for t in sorted_transitions]
        features = torch.tensor(features, dtype=torch.float)
        graph = LDBAGraph(features, edges, {
            0: 'root',
            **{i + 1: transition.positive_label for i, transition in enumerate(sorted_transitions)},
        }, root_assignments)
        return graph

    @classmethod
    def get_features(cls, transition: LDBATransition, possible_assignments: list[Assignment]) -> torch.tensor:
        # we use the following feature representation:
        # - 2^|propositions| features indicating which assignments satisfy the transition label
        # - 1 feature indicating whether the transition is an epsilon transition
        # - 1 feature indicating whether the transition is accepting
        features = []
        for assignment in possible_assignments:
            satisfies = assignment.to_frozen() in transition.valid_assignments
            features.append(int(satisfies))
        features.append(int(transition.is_epsilon()))
        features.append(int(transition.accepting))
        return features

    @classmethod
    def get_assignments(cls, transition: LDBATransition, possible_assignments: list[Assignment]) -> set[FrozenAssignment]:
        return {a.to_frozen() for a in possible_assignments if a.to_frozen() in transition.valid_assignments}

    @property
    def num_nodes(self):
        return self.x.shape[0]

    @property
    def num_edges(self):
        return self.edge_index.shape[1]
