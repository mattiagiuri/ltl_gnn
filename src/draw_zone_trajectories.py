import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from envs import make_env
from ltl import AvoidSampler, FixedSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search import ExhaustiveSearch
from utils.model_store import ModelStore
from visualize.zones import draw_trajectories

env_name = 'PointLtl2-v0'
exp = 'base'
seed = 1

random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

render = True
# sampler = FixedSampler.partial('GF yellow & GF blue')
# sampler = FixedSampler.partial('GF magenta & GF green & G (yellow => (!blue U magenta))')
sampler = AvoidSampler.partial(2, 1)
# sampler = FixedSampler.partial('(!magenta U yellow) & (!yellow U blue)')
# sampler = FixedSampler.partial('!(green | blue | yellow) U (magenta)')
deterministic = True

env = make_env(env_name, sampler, render_mode='human' if render else None, max_steps=1000)
config = model_configs['default']
model_store = ModelStore(env_name, exp, seed, None)
training_status = model_store.load_training_status(map_location='cpu')
model = build_model(env, training_status, config)

props = set(env.get_propositions())
search = ExhaustiveSearch(model, props, num_loops=2)
agent = Agent(model, search=search, propositions=props, verbose=render)

num_episodes = 8

trajectories = []
zone_poss = []

env.reset(seed=seed)

pbar = range(num_episodes)
if not render:
    pbar = tqdm(pbar)
for i in pbar:
    obs = env.reset()
    agent.reset()
    info = {'ldba_state_changed': True}
    done = False
    num_steps = 0

    zone_poss.append(env.zone_positions)
    agent_traj = []

    while not done:
        action = agent.get_action(obs, info, deterministic=deterministic)
        action = action.flatten()
        obs, reward, done, info = env.step(action)
        agent_traj.append(env.agent_pos[:2])
        num_steps += 1
        if done:
            trajectories.append(agent_traj)

env.close()
fig = draw_trajectories(zone_poss, trajectories, 4, 2)
plt.show()
fig.savefig('trajectories.pdf')
