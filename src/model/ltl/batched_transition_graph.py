import torch
from torch_geometric.data import Data

from model.ltl import TransitionGraph


class BatchedTransitionGraph:
    def __init__(self, transition_graphs: list[TransitionGraph], active_transitions: list[list[int]], device=None):
        self.tgs = transition_graphs
        self.active_transitions = active_transitions
        self.device = device

    def __getitem__(self, index) -> Data:
        """Returns a sub-batch of the given transition graphs. Incorporates the active transitions."""
        num_nodes = 0
        xs = []
        edge_indices = []
        active_transitions_edges = [[], []]
        for i, j in enumerate(index):
            tg = self.tgs[j]
            xs.append(tg.x)
            edge_indices.append(tg.edge_index + num_nodes)
            for t in self.active_transitions[j]:
                active_transitions_edges[0].append(t + num_nodes)
                active_transitions_edges[1].append(i)
            num_nodes += tg.num_nodes
        x = torch.cat(xs, dim=0)
        edge_index = torch.cat(edge_indices, dim=1)
        active_transitions_edges = torch.tensor(active_transitions_edges, dtype=torch.long)
        data = Data(
            x=x,
            edge_index=edge_index,
            active_transitions_edges=active_transitions_edges,
            first_nodes=torch.stack([x[0, :] for x in xs], dim=0),  # TODO: remove
            num_graphs=len(index)
        )
        assert data.validate()
        if self.device is not None:
            data = data.to(self.device)
        return data

    def all(self):
        return self[range(len(self.tgs))]
