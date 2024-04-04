import torch

from envs import make_env
from ltl import EventuallySampler
from model.model import build_model
from model.agent import Agent
from config import model_configs

env = make_env('PointLtl2-v0', EventuallySampler, render_mode='human')
config = model_configs['default']
training_status = torch.load('experiments/ppo/PointLtl2-v0/testseed/0/status.pth', map_location='cpu')
model = build_model(env, training_status, config)
agent = Agent(model)

obs = env.reset()
for i in range(5000):
    action = agent.get_action(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(obs['goal'])

    if done:
        obs = env.reset()

env.close()
