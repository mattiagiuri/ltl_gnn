import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from envs import make_env
from ltl import AvoidSampler, FixedSampler
from ltl.logic import Assignment
from ltl.samplers.reach_stay_sampler import ReachStaySampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from model.seq_agent import SequenceAgent
from sequence.samplers import sequence_samplers
from sequence.search import ExhaustiveSearch
from utils.model_store import ModelStore
from visualize.zones import draw_trajectories

env_name = 'PointLtl2-v0'
exp = 'eval'
seed = 2

random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

render = False
render_trajectories = True
sampler = ReachStaySampler.partial()
# sampler = FixedSampler.partial('FG magenta & G(! (yellow | green | blue))')
deterministic = False

env = make_env(env_name, sampler, render_mode='human' if render else None, max_steps=1000)
config = model_configs['default']
model_store = ModelStore('PointLtl2-v0', exp, seed, None)
training_status = model_store.load_training_status(map_location='cpu')
model = build_model(env, training_status, config)

props = set(env.get_propositions())
search = ExhaustiveSearch(model, props, num_loops=2)
agent = Agent(model, search=search, propositions=props, verbose=render)

num_episodes = 8 if render_trajectories else 500

trajectories = []
zone_poss = []

num_successes = 0
num_violations = 0
steps = []
rets = []
success_mask = []
props = []
stays = []

env.reset(seed=seed)

pbar = range(num_episodes)
if not render:
    pbar = tqdm(pbar)
for i in pbar:
    obs = env.reset()
    agent.reset()
    info = {'ldba_state_changed': True}
    if render:
       print(obs['goal'])
    done = False
    num_steps = 0

    zone_poss.append(env.zone_positions)
    agent_traj = []

    while not done:
        action = agent.get_action(obs, info, deterministic=deterministic)
        # if (action == -42).all():
        #     print('Took epsilon action')
        action = action.flatten()
        obs, reward, done, info = env.step(action)
        agent_traj.append(env.agent_pos[:2])
        if len(info['propositions']) > 0:
            props.append(list(info['propositions'])[0])
        num_steps += 1
        if done:
            trajectories.append(agent_traj)
            stays.append(info['num_accepting_visits'])
            if not render:
                pbar.set_postfix({
                    'med': np.median(stays),
                    'mean': np.mean(stays)
                })

env.close()
print(np.median(stays))
print(np.mean(stays))
if render_trajectories:
    fig = draw_trajectories(zone_poss, trajectories, 4, 2)
    plt.show()
