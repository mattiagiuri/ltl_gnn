import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GatedGraphConv, JumpingKnowledge
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
        self.jk = JumpingKnowledge(mode='max')
        # self.jk = JumpingKnowledge(mode='lstm', channels=embedding_dim, num_layers=num_layers)

    def forward(self, graph: Data | BatchedGraph) -> torch.tensor:
        if isinstance(graph, BatchedGraph):
            graph = graph.all
        assert graph.root_node_mask.dtype == torch.bool
        layer_embeds = []
        x = self.layers[0](graph.x, graph.edge_index)
        x = F.relu(x)
        layer_embeds.append(x)
        for i, layer in enumerate(self.layers[1:]):
            # prev = x
            if self.concat_initial_features:
                x = torch.cat((x, graph.x), dim=1)
            # x = prev + layer(x, graph.edge_index)
            x = layer(x, graph.edge_index)
            x = F.relu(x)
            layer_embeds.append(x)
        x = self.jk(layer_embeds)
        return x[graph.root_node_mask]
