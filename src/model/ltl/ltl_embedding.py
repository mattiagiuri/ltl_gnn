import torch
from torch import nn


class LtlEmbedding(nn.Module):
    def __init__(self,
                 num_states: int,
                 embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_states, embedding_dim)

    def forward(self, states: torch.tensor) -> torch.tensor:
        return self.embedding(states)
