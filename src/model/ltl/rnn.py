import torch
from torch import nn

from model.ltl.batched_sequence import BatchedSequence


class LDBARNN(nn.Module):
    def __init__(
            self,
            num_assignments: int,
            embedding_dim,
            num_layers: int,
            concat_initial_features: bool = True
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_assignments + 1, embedding_dim=embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1, batch_first=True)

    def forward(self, seq: tuple[torch.tensor, torch.tensor] | BatchedSequence) -> torch.tensor:
        if isinstance(seq, BatchedSequence):
            seq = seq.all()
        x, lens = seq
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(x)
        return h.squeeze(0)
