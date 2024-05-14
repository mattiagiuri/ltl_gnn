from typing import Type, Optional

from dataclasses import dataclass

from torch import nn


@dataclass
class ActorConfig:
    layers: list[int]
    activation: Optional[Type[nn.Module]] | dict[str, Type[nn.Module]]
    state_dependent_std: bool = False


@dataclass
class GNNConfig:
    embedding_dim: int
    num_layers: int
    concat_initial_features: bool = True


@dataclass
class StandardNetConfig:
    layers: list[int]
    activation: Optional[Type[nn.Module]]


@dataclass
class ModelConfig:
    actor: ActorConfig
    critic: StandardNetConfig
    gnn: GNNConfig
    env_net: Optional[StandardNetConfig]


default_model_config = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=dict(
            hidden=nn.ReLU,
            output=nn.Tanh
        ),
        state_dependent_std=True
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.Tanh
    ),
    gnn=GNNConfig(
        embedding_dim=16,
        num_layers=2,
        concat_initial_features=True
    ),
    env_net=StandardNetConfig(
        layers=[128, 64],
        activation=nn.Tanh
    )
)

pretraining = ModelConfig(
    actor=ActorConfig(
        layers=[],
        activation=None,
        state_dependent_std=False
    ),
    critic=StandardNetConfig(
        layers=[],
        activation=None
    ),
    gnn=GNNConfig(
        embedding_dim=16,
        num_layers=2,
        concat_initial_features=False
    ),
    env_net=None
)
