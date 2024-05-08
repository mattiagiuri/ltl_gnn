import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GatedGraphConv
import torch.nn.functional as F

from model.ltl.batched_graph import BatchedGraph


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
        gcn_conv_params = dict(add_self_loops=False, normalize=False)
        self.layers = nn.ModuleList([
            GCNConv(feature_dim, embedding_dim, **gcn_conv_params),
            *[GCNConv(embedding_dim + (feature_dim if concat_initial_features else 0), embedding_dim, **gcn_conv_params)
              for _ in range(num_layers - 1)]
        ])

    def forward(self, graph: Data | BatchedGraph) -> torch.tensor:
        if isinstance(graph, BatchedGraph):
            graph = graph.all
        assert graph.root_node_mask.dtype == torch.bool
        x = self.layers[0](graph.x, graph.edge_index)
        for i, layer in enumerate(self.layers[1:]):
            prev = x
            x = F.relu(x)
            if self.concat_initial_features:
                x = torch.cat((x, graph.x), dim=1)
            x = prev + layer(x, graph.edge_index)
        return x[graph.root_node_mask]
