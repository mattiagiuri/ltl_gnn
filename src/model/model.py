from typing import Any, Optional

import gymnasium
import torch
import torch.nn as nn

from config import ModelConfig
from model.ltl import LtlPosNegNet
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

    def forward(self, obs):
        env_embedding = self.env_net(obs.features) if self.env_net is not None else obs.features
        ltl_embedding = self.ltl_net(obs.pos_graph, obs.neg_graph)
        embedding = torch.cat([env_embedding, ltl_embedding], dim=1)

        dist = self.actor(embedding)
        value = self.critic(embedding).squeeze(1)
        return dist, value


def build_model(
        env: gymnasium.Env,
        training_status: dict[str, Any],
        model_config: ModelConfig,
        ltl_model_weights: Optional[dict],
        freeze_ltl_model: bool
) -> Model:
    obs_dim = env.observation_space['features'].shape[0]
    action_space = env.action_space
    action_dim = action_space.n if isinstance(action_space, gymnasium.spaces.Discrete) else action_space.shape[0]
    if model_config.env_net is None:
        env_net = None
        env_embedding_dim = obs_dim
    else:
        env_net = torch_utils.make_mlp_layers([obs_dim, *model_config.env_net.layers],
                                              activation=model_config.env_net.activation)
        env_embedding_dim = model_config.env_net.layers[-1]
    graph_feature_dim = env.observation_space['pos_graph'].node_space.shape[0]
    ltl_embedding_dim = 2 * model_config.gnn.embedding_dim
    ltl_net = LtlPosNegNet(graph_feature_dim, ltl_embedding_dim,
                           num_layers=model_config.gnn.num_layers,
                           concat_initial_features=model_config.gnn.concat_initial_features)
    if ltl_model_weights is not None:
        ltl_net.load_state_dict(ltl_model_weights)
    if freeze_ltl_model:
        for param in ltl_net.parameters():
            param.requires_grad = False

    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        actor = DiscreteActor(action_dim=action_dim,
                              layers=[env_embedding_dim + ltl_embedding_dim, *model_config.actor.layers],
                              activation=model_config.actor.activation)
    else:
        actor = ContinuousActor(action_dim=action_dim,
                                layers=[env_embedding_dim + ltl_embedding_dim, *model_config.actor.layers],
                                activation=model_config.actor.activation,
                                state_dependent_std=model_config.actor.state_dependent_std)

    critic = torch_utils.make_mlp_layers([env_embedding_dim + ltl_embedding_dim, *model_config.critic.layers, 1],
                                         activation=model_config.critic.activation,
                                         final_layer_activation=False)
    model = Model(actor, critic, ltl_net, env_net)
    if "model_state" in training_status:
        model.load_state_dict(training_status["model_state"])
    return model
