import safety_gymnasium
import torch
from torch import nn

from envs.ltl2action_wrapper import Ltl2ActionWrapper
from envs.ltl_wrapper import LtlWrapper
from envs.zones.safety_gym_wrapper.safety_gym_wrapper import SafetyGymWrapper
from model.ltl.ltl_embedding import LtlEmbedding
from model.policy.continuous_actor import ContinuousActor
from src.ltl_wrappers import LTLEnv
from model.model import Model
from model.agent import Agent
from utils import torch_utils

env = safety_gymnasium.make('PointLtl2-v0', render_mode="human")
env = SafetyGymWrapper(env)
env = LtlWrapper(env)
env = Ltl2ActionWrapper(env)
env = LTLEnv(env, 'full', None, 0.0)


def build_model():
    obs_dim = env.observation_space['features'].shape[0]
    action_dim = env.action_space.shape[0]
    env_embedding_dim = 64
    env_net = torch_utils.make_mlp_layers([obs_dim, 128, env_embedding_dim], activation=nn.Tanh)
    ltl_net = LtlEmbedding(5, 32)
    actor = ContinuousActor(action_dim=action_dim,
                            layers=[64 + 32, 64, 64, 64],
                            activation=dict(
                                hidden=nn.ReLU,
                                output=nn.Tanh
                            ),
                            state_dependent_std=True)
    critic = torch_utils.make_mlp_layers([64 + 32, 64, 64, 1], activation=nn.Tanh, final_layer_activation=False)
    return Model(env_net, ltl_net, actor, critic)


model_dir = 'storage/refactor/train/status.pt'
model = build_model()
model.load_state_dict(torch.load(model_dir)['model_state'])
agent = Agent(model)

obs = env.reset()

for i in range(5000):
    action = agent.get_action(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    print(obs['text'])

    if done:
        obs = env.reset()
        print(obs['text'])

env.close()
