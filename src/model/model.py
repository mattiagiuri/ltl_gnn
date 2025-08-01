from typing import Any, Optional

import gymnasium
import torch
import torch.nn as nn

from config import ModelConfig
from model.ltl.ltl_net import LTLNet
from model.ltl.ltl_net_gnn import LTLNetGNN
from model.ltl.ltl_net_transformer import LTLNetTransformer
from model.mixed_distribution import MixedDistribution
from preprocessing.vocab import VOCAB, var_names, assignment_vocab, augment_vars
from model.policy import ContinuousActor
from model.policy import DiscreteActor
from utils import torch_utils


class Model(nn.Module):
    def __init__(self,
                 actor: nn.Module,
                 critic: nn.Module,
                 ltl_net: nn.Module,
                 env_net: Optional[nn.Module],
                 ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.ltl_net = ltl_net
        self.env_net = env_net
        self.recurrent = False

    def compute_embedding(self, obs):
        env_embedding = self.env_net(obs.features) if self.env_net is not None else obs.features
        ltl_embedding = self.ltl_net(obs.seq)
        return torch.cat([env_embedding, ltl_embedding], dim=1)

    def forward(self, obs):
        embedding = self.compute_embedding(obs)
        dist = self.actor(embedding)
        dist.set_epsilon_mask(obs.epsilon_mask)
        value = self.critic(embedding).squeeze(1)
        return dist, value


def build_model(
        env: gymnasium.Env,
        training_status: dict[str, Any],
        model_config: ModelConfig,
) -> Model:
    if len(VOCAB) <= 3:
        raise ValueError('VOCAB not initialized')
    obs_shape = env.observation_space['features'].shape
    action_space = env.action_space
    action_dim = action_space.n if isinstance(action_space, gymnasium.spaces.Discrete) else action_space.shape[0]
    if model_config.env_net is not None:
        env_net = model_config.env_net.build(obs_shape)
        env_embedding_dim = env_net.embedding_size
    else:
        assert len(obs_shape) == 1
        env_net = None
        env_embedding_dim = obs_shape[0]

    if model_config.gnn_mode:
        # print(len(var_names))
        embedding = nn.Embedding(len(var_names) + 4, model_config.ltl_embedding_dim, padding_idx=0)
        ltl_net = LTLNetGNN(embedding, model_config.num_rnn_layers, model_config.num_gnn_layers, var_names, assignment_vocab, model_config.stay_mode)
        # print(ltl_net.embedding.weight)
    elif model_config.set_transformer:
        embedding = nn.Embedding(len(VOCAB), model_config.ltl_embedding_dim, padding_idx=VOCAB['PAD'])
        ltl_net = LTLNetTransformer(embedding, model_config.num_rnn_layers)
    else:
        embedding = nn.Embedding(len(VOCAB), model_config.ltl_embedding_dim, padding_idx=VOCAB['PAD'])
        ltl_net = LTLNet(embedding, model_config.set_net, model_config.num_rnn_layers)

    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        actor = DiscreteActor(action_dim=action_dim,
                              layers=[env_embedding_dim + ltl_net.embedding_dim, *model_config.actor.layers],
                              activation=model_config.actor.activation)
    else:
        actor = ContinuousActor(action_dim=action_dim,
                                layers=[env_embedding_dim + ltl_net.embedding_dim, *model_config.actor.layers],
                                activation=model_config.actor.activation,
                                state_dependent_std=model_config.actor.state_dependent_std)

    critic = torch_utils.make_mlp_layers([env_embedding_dim + ltl_net.embedding_dim, *model_config.critic.layers, 1],
                                         activation=model_config.critic.activation,
                                         final_layer_activation=False)

    model = Model(actor, critic, ltl_net, env_net)

    # print(training_status['model_state'])
    # print(model.ltl_net.embedding.weight)

    if "model_state" in training_status:
        model.load_state_dict(training_status["model_state"])
    return model


def build_model_gnn(
        env: gymnasium.Env,
        training_status: dict[str, Any],
        model_config: ModelConfig,
) -> Model:
    if len(VOCAB) <= 3 or len(var_names) == 0:
        raise ValueError('VOCAB not initialized')
    obs_shape = env.observation_space['features'].shape
    action_space = env.action_space
    action_dim = action_space.n if isinstance(action_space, gymnasium.spaces.Discrete) else action_space.shape[0]
    if model_config.env_net is not None:
        env_net = model_config.env_net.build(obs_shape)
        env_embedding_dim = env_net.embedding_size
    else:
        assert len(obs_shape) == 1
        env_net = None
        env_embedding_dim = obs_shape[0]

    embedding = nn.Embedding(len(var_names) + 4, model_config.ltl_embedding_dim, padding_idx=0)
    ltl_net = LTLNetGNN(embedding=embedding, num_rnn_layers=model_config.num_rnn_layers,
                        num_gnn_layers=model_config.num_gnn_layers, variable_names=var_names,
                        assignment_vocabulary=assignment_vocab)

    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        actor = DiscreteActor(action_dim=action_dim,
                              layers=[env_embedding_dim + ltl_net.embedding_dim, *model_config.actor.layers],
                              activation=model_config.actor.activation)
    else:
        actor = ContinuousActor(action_dim=action_dim,
                                layers=[env_embedding_dim + ltl_net.embedding_dim, *model_config.actor.layers],
                                activation=model_config.actor.activation,
                                state_dependent_std=model_config.actor.state_dependent_std)

    critic = torch_utils.make_mlp_layers([env_embedding_dim + ltl_net.embedding_dim, *model_config.critic.layers, 1],
                                         activation=model_config.critic.activation,
                                         final_layer_activation=False)

    model = Model(actor, critic, ltl_net, env_net)

    if "model_state" in training_status:
        model.load_state_dict(training_status["model_state"])
    return model