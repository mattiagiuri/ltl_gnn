import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import trange

from envs import make_env
from envs.flatworld import FlatWorld
from ltl import FixedSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search import ExhaustiveSearch
from utils.model_store import ModelStore
from visualize.zones import draw_trajectories

env_name = 'FlatWorld-v0'
exp = 'fixedregions'
seed = 1

random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

sampler = FixedSampler.partial('!(red | green) U magenta')
deterministic = False

env = make_env(env_name, sampler, render_mode=None, max_steps=50)
config = model_configs[env_name]
model_store = ModelStore(env_name, exp, seed, None)
training_status = model_store.load_training_status(map_location='cpu')
model = build_model(env, training_status, config)

props = set(env.get_propositions())
search = ExhaustiveSearch(model, props, num_loops=2)
agent = Agent(model, search=search, propositions=props, verbose=False)

num_episodes = 1

traj = []
obs, info = env.reset(), {}
traj.append(env.agent_pos)
agent.reset()
done = False

while not done:
    action = agent.get_action(obs, info, deterministic=deterministic)
    obs, reward, done, info = env.step(action)
    traj.append(env.agent_pos)
    if done:
        print(f'Success: {"success" in info}')
        break

FlatWorld.render(traj)
env.close()
