import torch

from envs import make_env
from ltl import EventuallySampler
from ltl.samplers.fixed_sampler import FixedSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs

env_name = 'PointLtl2-v0'
exp = 'gnn'
seed = 0

sampler = FixedSampler.partial_from_formula('GF blue & GF yellow & GF green')
env = make_env(env_name, sampler, render_mode='human')
config = model_configs['default']
training_status = torch.load(f'experiments/ppo/{env_name}/{exp}/{seed}/status.pth', map_location='cpu')
model = build_model(env, training_status, config, None, False)
agent = Agent(model)

obs = env.reset()
for i in range(5000):
    action = agent.get_action(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(obs['goal'])

    if done:
        obs = env.reset()

env.close()
