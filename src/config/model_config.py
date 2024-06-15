from abc import abstractmethod, ABC
from typing import Type, Optional

from dataclasses import dataclass

from torch import nn

from model.env import ConvEnvNet, StandardEnvNet


class AbstractModelConfig(ABC):
    @abstractmethod
    def build(self, input_shape: tuple[int, ...]) -> nn.Module:
        pass


@dataclass
class StandardNetConfig(AbstractModelConfig):
    layers: list[int]
    activation: Optional[Type[nn.Module]]

    def build(self, input_shape: tuple[int,]) -> nn.Module:
        return StandardEnvNet(input_shape[0], self.layers, self.activation)


@dataclass
class ConvNetConfig(AbstractModelConfig):
    channels: list[int]
    kernel_size: tuple[int, int]
    activation: Type[nn.Module]

    def build(self, input_shape: tuple[int, int, int]) -> nn.Module:
        return ConvEnvNet(input_shape, self.channels, self.kernel_size, self.activation)


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
class ModelConfig:
    actor: ActorConfig
    critic: StandardNetConfig
    gnn: GNNConfig
    env_net: Optional[AbstractModelConfig]


default = ModelConfig(
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

letter = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=nn.ReLU,
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
    env_net=ConvNetConfig(
        channels=[16, 32, 64],
        kernel_size=(2, 2),
        activation=nn.ReLU
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
