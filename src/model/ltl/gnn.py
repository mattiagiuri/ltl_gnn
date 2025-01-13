from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, ReLU
from torch_geometric.nn import GCNConv, GINConv


class GNN(nn.Module):

    def __init__(self, embedding_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([GCNConv(embedding_dim, embedding_dim) for _ in range(num_layers)])

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        return x


# class GNN(nn.Module):
#     def __init__(self, embedding_dim: int, num_layers: int):
#         super().__init__()
#         self.layers = nn.ModuleList([GINConv(nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
#                                                            BatchNorm1d(embedding_dim),
#                                                            Dropout(),
#                                                            ReLU(),
#                                                            nn.Linear(embedding_dim, embedding_dim),
#                                                            nn.ReLU()), train_eps=True)
#                                      for _ in range(num_layers)])
#
#     def forward(self, x, edge_index):
#         for layer in self.layers:
#             x = layer(x, edge_index)
#         return x
