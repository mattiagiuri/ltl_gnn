from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from utils import torch_utils


class DiscreteActor(nn.Module):
    def __init__(
            self,
            action_dim: int,
            layers: list[int],
            activation: Optional[nn.Module],
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = torch_utils.make_mlp_layers([*layers, action_dim], activation,
                                                 final_layer_activation=False)

    def forward(self, obs: torch.tensor) -> torch.distributions.Categorical:
        out = self.model(obs)
        return torch.distributions.Categorical(logits=out)
