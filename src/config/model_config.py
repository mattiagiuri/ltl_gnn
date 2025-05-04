from abc import abstractmethod, ABC
from typing import Type, Optional

from dataclasses import dataclass

from sympy.logic.boolalg import Boolean
from torch import nn

from model.env import ConvEnvNet, StandardEnvNet
from model.ltl.set_network import SetNetwork


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
class SetNetConfig(AbstractModelConfig):
    layers: list[int]
    activation: Type[nn.Module]

    def build(self, input_shape: int) -> nn.Module:
        return SetNetwork(input_shape, self.layers, self.activation)


@dataclass
class ActorConfig:
    layers: list[int]
    activation: Optional[Type[nn.Module]] | dict[str, Type[nn.Module]]
    state_dependent_std: bool = False


@dataclass
class ModelConfig:
    actor: ActorConfig
    critic: StandardNetConfig
    ltl_embedding_dim: int
    num_rnn_layers: int
    env_net: Optional[AbstractModelConfig]
    set_net: SetNetConfig
    gnn_mode: bool = False
    num_gnn_layers: int = 0
    freeze_gnn: bool = False
    stay_mode: bool = False
    set_transformer: bool = False


zones = ModelConfig(
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
    ltl_embedding_dim=16,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[128, 64],
        activation=nn.Tanh
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
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
    ltl_embedding_dim=32,
    num_rnn_layers=1,
    env_net=ConvNetConfig(
        channels=[16, 32, 64],
        kernel_size=(2, 2),
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 32],
        activation=nn.ReLU
    )
)

flatworld = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=16,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[16, 16],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    )
)

pretraining_context_flatworld = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=16,
    num_rnn_layers=1,
    env_net=None,
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn_mode=True,
    num_gnn_layers=5
)

pretraining_flatworld = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=16,
    num_rnn_layers=1,
    env_net=None,
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn_mode=True,
    num_gnn_layers=2
)


chessworld = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=16,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[16, 16],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    )
)


pretraining_chessworld = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=16,
    num_rnn_layers=1,
    env_net=None,
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn_mode=True,
    num_gnn_layers=2
)


chessworld_gnn = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=16,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[16, 16],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn_mode=True,
    num_gnn_layers=2,
    freeze_gnn=True
)


chessworld_gnn_train = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=16,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[16, 16],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn_mode=True,
    num_gnn_layers=2,
    freeze_gnn=False
)

chessworld_gnn_stay = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=16,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[16, 16],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn_mode=True,
    num_gnn_layers=3,
    freeze_gnn=True,
    stay_mode=True
)

pretraining_chessworld_stay = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=16,
    num_rnn_layers=1,
    env_net=None,
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn_mode=True,
    num_gnn_layers=3,
    stay_mode=True
)

chessworld_gnn_stay_fine = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=16,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[16, 16],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn_mode=True,
    num_gnn_layers=3,
    freeze_gnn=False,
    stay_mode=True
)

chessworld_gnn_big = ModelConfig(
    actor=ActorConfig(
        layers=[128, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[128, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=32,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn_mode=True,
    num_gnn_layers=3,
    freeze_gnn=False,
    stay_mode=True
)

chessworld_gnn_big_frozen = ModelConfig(
    actor=ActorConfig(
        layers=[128, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[128, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=32,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn_mode=True,
    num_gnn_layers=3,
    freeze_gnn=True,
    stay_mode=True
)

chessworld_deepsets_big = ModelConfig(
    actor=ActorConfig(
        layers=[128, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[128, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=32,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 32],
        activation=nn.ReLU
    ),
)

chessworld_transformer_big = ModelConfig(
    actor=ActorConfig(
        layers=[128, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[128, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=32,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 32],
        activation=nn.ReLU
    ),
    set_transformer=True
)

chessworld_transformer_frozen = ModelConfig(
    actor=ActorConfig(
        layers=[128, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[128, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=32,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 32],
        activation=nn.ReLU
    ),
    set_transformer=True,
    freeze_gnn=True
)

chessworld_gnn_big_prop = ModelConfig(
    actor=ActorConfig(
        layers=[128, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[128, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=32,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn_mode=True,
    num_gnn_layers=3,
    freeze_gnn=False,
    stay_mode=False
)

chessworld_gnn_big_frozen_prop = ModelConfig(
    actor=ActorConfig(
        layers=[128, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[128, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=32,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn_mode=True,
    num_gnn_layers=3,
    freeze_gnn=True,
    stay_mode=False
)

flatworld_gnn_big_frozen = ModelConfig(
    actor=ActorConfig(
        layers=[128, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[128, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=32,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn_mode=True,
    num_gnn_layers=3,
    freeze_gnn=True,
    stay_mode=True
)

flatworld_gnn_big = ModelConfig(
    actor=ActorConfig(
        layers=[128, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[128, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=32,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn_mode=True,
    num_gnn_layers=3,
    freeze_gnn=False,
    stay_mode=True
)

flatworld_deepsets_big = ModelConfig(
    actor=ActorConfig(
        layers=[128, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[128, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=32,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 32],
        activation=nn.ReLU
    )
)

flatworld_gnn = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=16,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[16, 16],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn_mode=True,
    num_gnn_layers=3,
    freeze_gnn=False,
    stay_mode=True
)

frozen_flatworld_gnn = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=16,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[16, 16],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn_mode=True,
    num_gnn_layers=3,
    freeze_gnn=False,
    stay_mode=True
)

zones_update = ModelConfig(
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
    ltl_embedding_dim=16,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[128, 64],
        activation=nn.Tanh
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn_mode=True,
    num_gnn_layers=4,
    freeze_gnn=False,
    stay_mode=True
)