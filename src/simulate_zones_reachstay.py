import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from envs import make_env
from ltl import AvoidSampler, FixedSampler
from ltl.logic import Assignment
from model.model import build_model
from model.agent import Agent
from config import model_configs
from model.seq_agent import SequenceAgent
from sequence.samplers import sequence_samplers
from sequence.search import ExhaustiveSearch
from utils.model_store import ModelStore
from visualize.zones import draw_trajectories

env_name = 'PointLtl2-v0'
exp = 'reachstay'
seed = 1

random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

render = True
# sampler = AvoidSampler.partial(2, 1)
# sampler = FixedSampler.partial('G green')
props = ['green', 'magenta', 'blue', 'yellow']
reach_any = frozenset([
    Assignment.zero_propositions(props).to_frozen(),
    *[Assignment.single_proposition(p, props).to_frozen() for p in props]
])
print(len(reach_any))
prop = 'green'
reach_green = frozenset([Assignment.single_proposition(prop, props).to_frozen()])
avoid_none = frozenset()
avoid_others = frozenset([
    Assignment.zero_propositions(props).to_frozen(),
    *[Assignment.single_proposition(p, props).to_frozen() for p in props if p != prop]
])
seq = [(reach_green, avoid_others)] * 3
seq2 = [(reach_green, avoid_none), *[(reach_green, avoid_others)] * 2]
sampler = sequence_samplers.fixed(tuple(seq))
deterministic = True

env = make_env(env_name, sampler, render_mode='human' if render else None, max_steps=1000, sequence=True)
config = model_configs['default']
model_store = ModelStore('PointLtl2-v0', exp, seed, None)
training_status = model_store.load_training_status(map_location='cpu')
model = build_model(env, training_status, config)

search = ExhaustiveSearch(model, num_loops=2)
agent = SequenceAgent(model, verbose=render)

num_episodes = 8

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
    # if render:
    #    print(obs['goal'])
    done = False
    num_steps = 0

    zone_poss.append(env.zone_positions)
    agent_traj = []

    while not done:
        action = agent.get_action(obs, info, deterministic=deterministic)
        action = action.flatten()

        value = agent.get_value(obs, seq)
        old_value = agent.get_value(obs, seq2)
        print(value, old_value)
        if value >= 0.85 and not env.switched:
            if render:
                print('SWITCHED: ', value)
            env.switch()

        obs, reward, done, info = env.step(action)
        agent_traj.append(env.agent_pos[:2])
        if len(info['propositions']) > 0:
            props.append(list(info['propositions'])[0])
        num_steps += 1
        if done:
            trajectories.append(agent_traj)
            stays.append(info['max_stay'])
            if 'success' in info:
                num_successes += 1
                final_reward = 1
                steps.append(num_steps)
            elif 'violation' in info:
                num_violations += 1
                final_reward = -1
            else:
                final_reward = 0
            # print(final_reward)
            rets.append(final_reward * 0.998 ** (num_steps - 1))
            success_mask.append('success' in info)
            if not render:
                pbar.set_postfix({
                    'MS': np.median(stays)
                })

env.close()
print(np.median(stays))
print(np.mean(stays))
# fig = draw_trajectories(zone_poss, trajectories, 4, 2)
# plt.show()
