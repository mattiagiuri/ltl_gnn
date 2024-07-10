import torch
from torch import nn

from preprocessing.batched_sequence import BatchedSequence


class ReachAvoidRNN(nn.Module):
    def __init__(
            self,
            num_assignments: int,
            embedding_dim,
            num_layers: int,
    ):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("Embedding dimension must be even.")
        # + padding + empty
        self.embedding = nn.Embedding(num_embeddings=num_assignments + 2, embedding_dim=embedding_dim // 2, padding_idx=0)
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=num_layers, batch_first=True)
        self.embedding_size = embedding_dim

    def forward(self, seq: tuple[torch.tensor, torch.tensor, torch.tensor] | BatchedSequence) -> torch.tensor:
        if isinstance(seq, BatchedSequence):
            seq = seq.all()
        reach, avoid, lens = seq
        reach = self.embedding(reach)
        avoid = self.embedding(avoid)
        x = torch.cat([reach, avoid], dim=-1)
        seq = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(seq)
        return h[-1, ...]
