import copy
from typing import Any, Optional

import gymnasium
import torch
import torch.nn as nn

from config import ModelConfig
from model.ltl.rnn import LDBARNN
from model.policy import ContinuousActor
from model.policy import DiscreteActor
from utils import torch_utils


class Model(nn.Module):
    def __init__(self,
                 actor: nn.Module,
                 critic: nn.Module,
                 q_net: nn.Module,
                 ltl_net: nn.Module,
                 env_net: Optional[nn.Module],
                 ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.q_net = q_net
        self.ltl_net = ltl_net
        self.env_net = env_net
        self.recurrent = False

    def compute_embedding(self, obs):
        env_embedding = self.env_net(obs.features) if self.env_net is not None else obs.features
        # ltl_embedding = self.ltl_net(obs.pos_graph, obs.neg_graph)
        ltl_embedding = self.ltl_net(obs.seq)
        return torch.cat([env_embedding, ltl_embedding], dim=1)

    def forward(self, obs):
        embedding = self.compute_embedding(obs)
        dist = self.actor(embedding)
        value = self.critic(embedding).squeeze(1)
        return dist, value

    def forward_q(self, obs, action):
        embedding = self.compute_embedding(obs)
        # TODO: split into continuous and discrete module
        action = torch.nn.functional.one_hot(action.long(), num_classes=4)
        embedding = torch.cat([embedding, action], dim=1)
        q_value = self.q_net(embedding).squeeze(1)
        return q_value


def build_model(
        env: gymnasium.Env,
        training_status: dict[str, Any],
        model_config: ModelConfig,
) -> Model:
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
    # graph_feature_dim = env.observation_space['pos_graph'].node_space.shape[0]
    # ltl_embedding_dim = 2 * model_config.gnn.embedding_dim
    # ltl_net = LtlPosNegNet(graph_feature_dim, ltl_embedding_dim,
    #                        num_layers=model_config.gnn.num_layers,
    #                        concat_initial_features=model_config.gnn.concat_initial_features)

    # if ltl_model_weights is not None:
    #     ltl_net.load_state_dict(ltl_model_weights)
    # if freeze_ltl_model:
    #     for param in ltl_net.parameters():
    #         param.requires_grad = False

    ltl_embedding_dim = 64
    # num_assignments = 4
    num_assignments = 12
    ltl_net = LDBARNN(num_assignments, ltl_embedding_dim, num_layers=1)  # TODO: careful, parameterise this!
    print(torch_utils.get_number_of_params(ltl_net))

    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        actor = DiscreteActor(action_dim=action_dim,
                              layers=[env_embedding_dim + ltl_net.embedding_size, *model_config.actor.layers],
                              activation=model_config.actor.activation)
    else:
        actor = ContinuousActor(action_dim=action_dim,
                                layers=[env_embedding_dim + ltl_net.embedding_size, *model_config.actor.layers],
                                activation=model_config.actor.activation,
                                state_dependent_std=model_config.actor.state_dependent_std)

    critic = torch_utils.make_mlp_layers([env_embedding_dim + ltl_net.embedding_size, *model_config.critic.layers, 1],
                                         activation=model_config.critic.activation,
                                         final_layer_activation=False)

    q_net = torch_utils.make_mlp_layers([env_embedding_dim + ltl_net.embedding_size + action_dim, *model_config.critic.layers, 1],
                                         activation=model_config.critic.activation,
                                         final_layer_activation=False)

    model = Model(actor, critic, q_net, ltl_net, env_net)

    if "model_state" in training_status:
        updated_training_status = copy.deepcopy(training_status)
        for key in training_status["model_state"]:
            if key.startswith("env_net") and not (key.startswith("env_net.mlp") or key.startswith("env_net.conv")):
                weights = training_status["model_state"][key]
                del updated_training_status["model_state"][key]
                updated_training_status["model_state"][f"env_net.mlp.{key[len('env_net.'):]}"] = weights
        q_keys = ["q_net.0.weight", "q_net.0.bias", "q_net.2.weight", "q_net.2.bias", "q_net.4.weight", "q_net.4.bias"]
        current_state_dict = model.state_dict()
        for key in q_keys:
            if key not in training_status["model_state"]:
                updated_training_status["model_state"][key] = current_state_dict[key]
        model.load_state_dict(updated_training_status["model_state"])
    return model
