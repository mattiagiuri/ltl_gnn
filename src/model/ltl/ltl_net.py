import torch
from torch import nn

from preprocessing import BatchedSequences, BatchedASTSequence
from model.ltl.gnn import GNN


class LTLNet(nn.Module):
    def __init__(
            self,
            embedding: nn.Module,
            num_gnn_layers: int,
            num_rnn_layers: int,
    ):
        super().__init__()
        self.embedding = embedding
        embedding_dim = embedding.embedding_dim

        self.gnn = GNN(embedding_dim, num_gnn_layers)
        self.rnn = nn.GRU(input_size=2 * embedding_dim, hidden_size=2 * embedding_dim, num_layers=num_rnn_layers,
                          batch_first=True)
        self.embedding_dim = 2 * embedding_dim

    def forward(self, batched_seqs: tuple[BatchedSequences, BatchedSequences] | BatchedASTSequence) -> torch.tensor:
        if isinstance(batched_seqs, BatchedASTSequence):
            reach_seq, avoid_seq = batched_seqs.all()
        else:
            reach_seq, avoid_seq = batched_seqs
        assert (reach_seq.lens == avoid_seq.lens).all()
        reach = self.process_sequence(reach_seq)
        avoid = self.process_sequence(avoid_seq)
        x = torch.cat([reach, avoid], dim=-1)
        seq = nn.utils.rnn.pack_padded_sequence(x, reach_seq.lens, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(seq)
        return h[-1, ...]

    def process_sequence(self, seq: BatchedSequences) -> torch.tensor:  # returns # (B, L, D)
        batched_embedding = self.embedding(seq.batched)
        if seq.graph_indices.sum() == 0:
            return batched_embedding
        x = self.embedding(seq.batched_graph.x)
        graph_embedding = self.gnn(x, seq.batched_graph.edge_index)
        batched_embedding[seq.graph_indices] = graph_embedding[seq.batched_graph.root_indices]
        return batched_embedding
