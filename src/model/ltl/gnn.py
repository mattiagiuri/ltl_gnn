import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from model.ltl.batched_transition_graph import BatchedTransitionGraph


class GNN(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            embedding_dim,
            num_layers: int,
            concat_initial_features: bool = True
    ):
        super().__init__()
        self.concat_initial_features = concat_initial_features
        # self.layers = nn.ModuleList([
        #     GCNConv(feature_dim, embedding_dim),  # TODO: self loops?
        #     *[GCNConv(embedding_dim + (feature_dim if concat_initial_features else 0), embedding_dim)
        #       for _ in range(num_layers - 1)]
        # ])
        self.conv1 = GCNConv(7, embedding_dim)
        # self.conv2 = GCNConv(embedding_dim, embedding_dim)
        # self.embedding = nn.Embedding(9, embedding_dim)

    def forward(self, tg: Data | BatchedTransitionGraph) -> torch.tensor:
        if isinstance(tg, BatchedTransitionGraph):
            tg = tg.all()
        x = tg.x
        # x = self.embedding(x[:, 0]) # + self.accepting_embedding(x[:, 1])
        x = self.conv1(x, tg.edge_index)
        # x = F.relu(x)
        # x = self.conv2(x, tg.edge_index)
        # for i, layer in enumerate(self.layers):
        #     if i != 0 and self.concat_initial_features:
        #         x = torch.cat((x, tg.x), dim=1)
        #     x = layer(x, tg.edge_index)
        #     if i != len(self.layers) - 1:
        #         x = F.relu(x)

        # aggregate embeddings of active transitions
        embeddings = torch.zeros(tg.num_graphs, x.shape[1], device=x.device)
        edge_index = tg.active_transitions_edges
        scatter_index = edge_index[1].unsqueeze(1).expand(edge_index.shape[1], x.shape[1])
        embeddings = embeddings.scatter_add(0, scatter_index, x[edge_index[0]])  # [num_graphs, embedding_dim]
        return embeddings
