from typing import Any

import gymnasium
import torch
import torch.nn as nn

from config import ModelConfig
from model.ltl import LtlEmbedding
from model.policy import ContinuousActor
from utils import torch_utils


class Model(nn.Module):
    def __init__(self,
                 env_net: nn.Module,
                 ltl_net: nn.Module,
                 actor: nn.Module,
                 critic: nn.Module
                 ):
        super().__init__()
        self.env_net = env_net
        self.ltl_net = ltl_net
        self.actor = actor
        self.critic = critic
        self.recurrent = False

    def forward(self, obs):
        env_embedding = self.env_net(obs.features)
        ltl_embedding = self.ltl_net(obs.goal)
        embedding = torch.cat([env_embedding, ltl_embedding], dim=1)

        dist = self.actor(embedding)
        value = self.critic(embedding).squeeze(1)
        return dist, value


def build_model(env: gymnasium.Env, training_status: dict[str, Any], model_config: ModelConfig) -> Model:
    obs_dim = env.observation_space['features'].shape[0]
    action_dim = env.action_space.shape[0]
    env_net = torch_utils.make_mlp_layers([obs_dim, *model_config.env_net.layers],
                                          activation=model_config.env_net.activation)
    env_embedding_dim = model_config.env_net.layers[-1]
    ltl_embedding_dim = 32
    ltl_net = LtlEmbedding(5, ltl_embedding_dim)
    actor = ContinuousActor(action_dim=action_dim,
                            layers=[env_embedding_dim + ltl_embedding_dim, *model_config.actor.layers],
                            activation=model_config.actor.activation,
                            state_dependent_std=model_config.actor.state_dependent_std)
    critic = torch_utils.make_mlp_layers([env_embedding_dim + ltl_embedding_dim, *model_config.critic.layers, 1],
                                         activation=model_config.critic.activation,
                                         final_layer_activation=False)
    model = Model(env_net, ltl_net, actor, critic)
    if "model_state" in training_status:
        model.load_state_dict(training_status["model_state"])
    return model
