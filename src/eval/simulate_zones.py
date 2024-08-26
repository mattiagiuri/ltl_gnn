import random

import numpy as np
import torch
from tqdm import tqdm

from envs import make_env
from ltl import AvoidSampler, FixedSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search import ExhaustiveSearch
from utils.model_store import ModelStore

env_name = 'PointLtl2-v0'
exp = 'punishwall'
seed = 2

random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

render = False
# sampler = AvoidSampler.partial(2, 1)
# sampler = FixedSampler.partial('(!magenta U yellow) & (!yellow U blue)')
sampler = FixedSampler.partial('!(green | blue | yellow) U (magenta)')
deterministic = True

env = make_env(env_name, sampler, render_mode='human' if render else None, max_steps=1000)
config = model_configs['default']
model_store = ModelStore(env_name, exp, seed, None)
training_status = model_store.load_training_status(map_location='cpu')
model = build_model(env, training_status, config)

props = set(env.get_propositions())
search = ExhaustiveSearch(model, props, num_loops=2)
agent = Agent(model, search=search, propositions=props, verbose=render)

num_episodes = 500

num_successes = 0
num_violations = 0
steps = []
rets = []
success_mask = []
props = []

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
    while not done:
        action = agent.get_action(obs, info, deterministic=deterministic)
        action = action.flatten()
        obs, reward, done, info = env.step(action)
        if len(info['propositions']) > 0:
            props.append(list(info['propositions'])[0])
        num_steps += 1
        if done:
            if 'success' in info:
                num_successes += 1
                final_reward = 1
                steps.append(num_steps)
            elif 'violation' in info:
                num_violations += 1
                final_reward = -1
            else:
                final_reward = 0
            rets.append(final_reward * 0.998 ** (num_steps - 1))
            success_mask.append('success' in info)
            if not render:
                pbar.set_postfix({
                    's': num_successes / (i + 1),
                    'v': num_violations / (i + 1),
                    'ADR(t)': np.mean(rets),
                    'ADR(s)': np.mean(np.array(rets)[success_mask]),
                    'AS': np.mean(steps),
                    'AS(m)': np.median(steps),
                })

env.close()
print(f'Success rate: {num_successes / num_episodes}')
print(f'Violation rate: {num_violations / num_episodes}')
print(f'Num total: {num_episodes}')
print(f'ADR (total): {np.mean(rets):.3f}')
print(f'ADR (successful): {np.mean(np.array(rets)[success_mask]):.3f}')
print(f'AS: {np.mean(steps):.3f}')
print(f'AS (median): {np.median(steps):.3f}')

print(
    f'{num_successes / num_episodes:.3f},{num_violations / num_episodes:.3f},{np.mean(rets):.3f},{np.mean(np.array(rets)[success_mask]):.3f},{np.mean(steps):.3f},{np.median(steps):.3f}')
