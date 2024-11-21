import torch
from torch import nn

from model.ltl.set_network import SetNetwork
from preprocessing import BatchedReachAvoidSequences, BatchedSequences
from model.ltl.gnn import GNN


class LTLNetGNN(nn.Module):
    def __init__(
            self,
            embedding: nn.Module,
            num_rnn_layers: int,
            num_gnn_layers: int
    ):
        super().__init__()
        self.embedding = embedding
        embedding_dim = embedding.embedding_dim

        self.gnn = GNN(embedding_dim=embedding_dim, num_layers=num_gnn_layers)
        self.rnn = nn.GRU(input_size=2 * embedding_dim, hidden_size=2 * embedding_dim, num_layers=num_rnn_layers,
                          batch_first=True)

        self.embedding_dim = 2 * embedding_dim

    def forward(self, batched_seqs: tuple[tuple[torch.tensor, torch.tensor], tuple[torch.tensor, torch.tensor]]
                                    | BatchedReachAvoidSequences, edge_index: torch.tensor) -> torch.tensor:
        if isinstance(batched_seqs, BatchedReachAvoidSequences):
            (reach_lens, reach_data), (avoid_lens, avoid_data) = batched_seqs.all()
        else:
            (reach_lens, reach_data), (avoid_lens, avoid_data) = batched_seqs
        assert (reach_lens == avoid_lens).all()

        """TODO add edge indices properly"""
        reach = self.embedding(reach_data)
        reach = self.gnn(reach, edge_index)

        avoid = self.embedding(avoid_data)
        avoid = self.gnn(avoid, edge_index)

        x = torch.cat([reach, avoid], dim=-1)
        seq = nn.utils.rnn.pack_padded_sequence(x, reach_lens, batch_first=True, enforce_sorted=False)

        _, h = self.rnn(seq)
        return h[-1, ...]
