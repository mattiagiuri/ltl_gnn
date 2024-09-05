import pickle
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from envs import make_env
from ltl import AvoidSampler, FixedSampler
from ltl.samplers.reach_sampler import ReachSampler
from ltl.samplers.super_sampler import SuperSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search import ExhaustiveSearch
from utils.model_store import ModelStore
from visualize.zones import draw_trajectories, draw_multiple_trajectories

env_name = 'PointLtl2-v0'
exp = 'eval'
seed = 2

random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

render = False
# sampler = FixedSampler.partial('(! blue U green) | F yellow')
# sampler = FixedSampler.partial('F yellow')
# sampler = FixedSampler.partial('GF magenta & GF green & G (yellow => (!blue U magenta))')
# sampler = AvoidSampler.partial(2, 1)
# sampler = FixedSampler.partial('(!magenta U yellow) & (!yellow U blue)')
# sampler = FixedSampler.partial('!green U (yellow & (!magenta U blue))')
# sampler = FixedSampler.partial('((yellow => F magenta) U green) & F blue')
avoid_sampler = AvoidSampler.partial((1, 2), 1)
reach_sampler = ReachSampler.partial((1, 3))
sampler = SuperSampler.partial(reach_sampler, avoid_sampler)
deterministic = True

# 1: !green U (yellow & (!magenta U blue))

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

env.reset(seed=3)  # settings for nice GF yellow & GF blue trajectory: exp eval, seed 5, reset seed 6

pbar = range(num_episodes)
if not render:
    pbar = tqdm(pbar)
for i in pbar:
    obs = env.reset()
    print(obs['goal'])
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
# traj = trajectories[1]
# with open('fail_optimal.traj', 'wb') as f:
#     pickle.dump(traj, f)
# with open('fail_safety.traj', 'rb') as f:
#     traj2 = pickle.load(f)
# with open('fail_optimal.traj', 'rb') as f:
#     traj2 = pickle.load(f)
# trajs = [trajectories[0], traj2]
# with open('tmp.pz', 'wb') as f:
#     pickle.dump(traj, f)
# fig = draw_multiple_trajectories(zone_poss[0], trajs, ['solid', 'dashed'], ['green', 'orange'])
cols = 4 if len(zone_poss) > 4 else len(zone_poss)
rows = 1 if len(zone_poss) <= 4 else 2
fig = draw_trajectories(zone_poss, trajectories, cols, rows)
plt.show()
