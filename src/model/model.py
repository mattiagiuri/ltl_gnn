from typing import Any, Optional

import gymnasium
import torch
import torch.nn as nn

from config import ModelConfig
from model.ltl import LtlEmbedding
from model.ltl.gnn import GNN
from model.policy import ContinuousActor
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
        ltl_embedding = self.ltl_net(obs.transition_graph)
        embedding = torch.cat([env_embedding, ltl_embedding], dim=1)

        dist = self.actor(embedding)
        value = self.critic(embedding).squeeze(1)
        return dist, value


def build_model(env: gymnasium.Env, training_status: dict[str, Any], model_config: ModelConfig) -> Model:
    obs_dim = env.observation_space['features'].shape[0]
    action_dim = env.action_space.shape[0]
    if model_config.env_net is None:
        env_net = None
        env_embedding_dim = obs_dim
    else:
        env_net = torch_utils.make_mlp_layers([obs_dim, *model_config.env_net.layers],
                                              activation=model_config.env_net.activation)
        env_embedding_dim = model_config.env_net.layers[-1]
    # ltl_embedding_dim = 4
    # ltl_net = LtlEmbedding(5, ltl_embedding_dim)
    ltl_embedding_dim = 16
    gnn_feature_dim = -1 # env.observation_space['transition_graph'].node_space.shape[0]
    ltl_net = GNN(gnn_feature_dim, ltl_embedding_dim, num_layers=2, concat_initial_features=False)
    print(f'Num GNN parameters: {torch_utils.get_number_of_params(ltl_net)}')
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
