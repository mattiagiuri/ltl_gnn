import torch
from torch_geometric.data import Data

from ltl.automata.ldba import LDBA, LDBATransition
from ltl.logic.assignment import Assignment


class TransitionGraph(Data):

    def __init__(self, edge_index: torch.tensor, features: torch.tensor, labels: dict[int, str], **kwargs):
        super().__init__(x=features, edge_index=edge_index, **kwargs)
        self.labels = labels

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
        for state in range(ldba.num_states):
            for transition in ldba.state_to_transitions[state]:
                if transition not in transition_to_index:
                    transition_to_index[transition] = len(transition_to_index)
                current_index = transition_to_index[transition]
                for incoming_transition in ldba.state_to_incoming_transitions[state]:
                    if incoming_transition not in transition_to_index:
                        transition_to_index[incoming_transition] = len(transition_to_index)
                    edge_index.append((current_index, transition_to_index[incoming_transition]))
                labels[current_index] = transition.label
                features[current_index] = TransitionGraph.get_features(transition)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        assert all(feature is not None for feature in features)
        features = torch.stack(features, dim=0)
        return TransitionGraph(edge_index, features, labels)

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

    def display(self):
        # TODO: make accepting transitions green, make epsilon transitions dashed, make sink transition red
        pass
