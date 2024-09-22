from typing import Type

import torch
from torch import nn
import torch.nn.functional as F


class SetNetwork(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, activation: Type[nn.Module] = nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x.sum(dim=-2)
        x = self.linear(x)
        return self.activation(x)
