import torch
from torch import nn
from torch_geometric.data import Data

from model.ltl.batched_graph import BatchedGraph
from model.ltl.gnn import GNN


class LtlPosNegNet(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            embedding_dim,
            num_layers: int,
            concat_initial_features: bool = True
    ):
        super().__init__()
        if not embedding_dim % 2 == 0:
            raise ValueError('Embedding dimension must be even.')
        embedding_dim //= 2
        self.pos_gnn = GNN(feature_dim, embedding_dim, num_layers, concat_initial_features)
        # self.neg_gnn = GNN(feature_dim, embedding_dim, num_layers, concat_initial_features)
        # self.layer = nn.Linear(2 * embedding_dim, 2 * embedding_dim)

    def forward(self, pos_graph: Data | BatchedGraph, neg_graph: Data | BatchedGraph) -> torch.tensor:
        pos_embedding = self.pos_gnn(pos_graph)
        neg_embedding = self.pos_gnn(neg_graph)
        return torch.cat([pos_embedding, neg_embedding], dim=1)

        # neg_embedding = self.neg_gnn(neg_graph)
        # cat = torch.cat([pos_embedding, neg_embedding], dim=1)
        # cat = relu(cat)
        # return relu(self.layer(cat))
