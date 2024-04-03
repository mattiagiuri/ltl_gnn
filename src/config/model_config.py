from typing import Type

from dataclasses import dataclass

from torch import nn


@dataclass
class ActorConfig:
    layers: list[int]
    activation: Type[nn.Module] | dict[str, Type[nn.Module]]
    state_dependent_std: bool = False


@dataclass
class StandardNetConfig:
    layers: list[int]
    activation: Type[nn.Module]


@dataclass
class ModelConfig:
    actor: ActorConfig
    critic: StandardNetConfig
    env_net: StandardNetConfig


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
    env_net=StandardNetConfig(
        layers=[128, 64],
        activation=nn.Tanh
    )
)
