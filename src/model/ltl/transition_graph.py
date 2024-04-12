from dataclasses import dataclass

import torch
from torch_geometric.data import Data

from ltl.automata import LDBA, LDBATransition
from ltl.logic import Assignment


class TransitionGraph(Data):
    @dataclass
    class Info:
        labels: dict[int, str]
        accepting_transitions: set[int]
        epsilon_transitions: set[int]
        sink_transitions: set[int]

    def __init__(self, edge_index: torch.tensor, features: torch.tensor, info: 'TransitionGraph.Info', **kwargs):
        super().__init__(x=features, edge_index=edge_index, **kwargs)
        self.info = info

    @staticmethod
    def from_ldba(ldba: LDBA) -> 'TransitionGraph':
        if not ldba.complete:
            raise ValueError(
                'LDBA must be complete. Make sure to call `complete_sink_state` before creating the transition graph.'
            )
        transition_to_index = {}
        edge_index = []
        features = [None] * ldba.num_transitions
        labels = {}
        accepting_transitions = set()
        epsilon_transitions = set()
        sink_transitions = set()
        for state in range(ldba.num_states):
            for transition in ldba.state_to_transitions[state]:
                if transition not in transition_to_index:
                    transition_to_index[transition] = len(transition_to_index)
                current_index = transition_to_index[transition]
                for incoming_transition in ldba.state_to_incoming_transitions[state]:
                    if incoming_transition not in transition_to_index:
                        transition_to_index[incoming_transition] = len(transition_to_index)
                    edge_index.append((current_index, transition_to_index[incoming_transition]))
                features[current_index] = TransitionGraph.get_features(transition)
                labels[current_index] = transition.label
                if transition.accepting:
                    accepting_transitions.add(current_index)
                if transition.is_epsilon():
                    epsilon_transitions.add(current_index)
                if transition.target == ldba.sink_state:
                    sink_transitions.add(current_index)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        assert all(feature is not None for feature in features)
        features = torch.stack(features, dim=0)
        info = TransitionGraph.Info(labels, accepting_transitions, epsilon_transitions, sink_transitions)
        return TransitionGraph(edge_index, features, info)

    @staticmethod
    def get_features(transition: LDBATransition) -> torch.tensor:
        # we use the following feature representation:
        # - 2^|propositions| features indicating which assignments satisfy the transition label
        # - 1 feature indicating whether the transition is an epsilon transition
        # - 1 feature indicating whether the transition is accepting
        features = []
        for assignment in Assignment.all_possible_assignments(transition.propositions):
            satisfies = assignment.to_frozen() in transition.valid_assignments
            features.append(int(satisfies))
        features.append(int(transition.is_epsilon()))
        features.append(int(transition.accepting))
        return torch.tensor(features, dtype=torch.float)
