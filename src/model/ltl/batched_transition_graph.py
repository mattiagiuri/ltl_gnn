import functools

import torch
from torch_geometric.data import Data

from model.ltl import TransitionGraph


class BatchedTransitionGraph:
    def __init__(self, transition_graphs: list[TransitionGraph], device=None):
        self.batch_x, self.batch_edge_index = self.batch(transition_graphs, device)
        self.batch_num_nodes = torch.tensor([tg.num_nodes for tg in transition_graphs], dtype=torch.long)

    @staticmethod
    def batch(transition_graphs: list[TransitionGraph], device=None) -> tuple[torch.tensor, torch.tensor]:
        max_nodes = max(tg.num_nodes for tg in transition_graphs)
        max_edges = max(tg.num_edges for tg in transition_graphs)
        feature_dim = transition_graphs[0].x.shape[1]
        batch_size = len(transition_graphs)
        x = torch.zeros((batch_size, max_nodes, feature_dim), dtype=torch.float)
        edge_index = torch.zeros((batch_size, 2, max_edges), dtype=torch.long)
        for i, tg in enumerate(transition_graphs):
            x[i, :tg.num_nodes, :] = tg.x
            edge_index[i, :, :tg.num_edges] = tg.edge_index
            edge_index[i, :, tg.num_edges:] = tg.num_nodes  # padding
        return x.to(device), edge_index.to(device)

    def __getitem__(self, index) -> Data:
        """Returns a sub-batch of the given transition graphs."""
        x = self.batch_x[index].reshape(-1, self.batch_x.shape[2])
        edge_index = self.batch_edge_index[index]
        # Shift the edges of subsequent graphs in the batch
        cumsum = self.batch_num_nodes[index].cumsum(0) - self.batch_num_nodes[index[0]]
        edge_index += cumsum.reshape(-1, 1, 1)
        edge_index = edge_index.transpose(0, 1).reshape(2, -1).contiguous()
        data = Data(
            x=x,
            edge_index=edge_index,
            num_graphs=len(index)
        )
        assert data.validate()
        return data

    @functools.cached_property
    def all(self):
        return self[range(self.batch_x.shape[0])]
