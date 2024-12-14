import torch
import torch_geometric
import torch.nn.functional as F


class GNN(torch.nn.Module):

    def __init__(self, embedding_dim: int, num_layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch_geometric.nn.GCNConv(embedding_dim, embedding_dim) for _ in range(num_layers)])

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        return x

gnn = GNN(4, 2)
embedding = torch.tensor([[1, 0, 0, 1]], dtype=torch.float)
edge_index = torch.tensor([[], []], dtype=torch.long)


print(gnn(embedding, edge_index))
