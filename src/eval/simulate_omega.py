import random

import numpy as np
import torch
from pandas.io.pytables import Fixed
from tqdm import tqdm

from envs import make_env
from ltl import AvoidSampler, FixedSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search import ExhaustiveSearch
from utils.model_store import ModelStore

env_name = 'PointLtl2-v0'
exp = 'eval'
seed = 1

random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

render = False
# sampler = AvoidSampler.partial(2, 1)
# sampler = FixedSampler.partial('(!magenta U yellow) & (!yellow U blue)')
# sampler = FixedSampler.partial('!(green | blue | yellow) U (magenta)')
# sampler = FixedSampler.partial('(yellow => F magenta) U green')
# sampler = FixedSampler.partial('(F green | F yellow) & G ! blue')
# sampler = FixedSampler.partial('(!(green | blue)) U (yellow | magenta)')
sampler = FixedSampler.partial('GF blue & GF green & GF yellow')
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
total_omega = 0

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
            total_omega += info['num_accepting_visits']
            if not render:
                pbar.set_postfix({
                    'omega': total_omega / (i + 1),
                })

env.close()
print(f'Omega: {total_omega / num_episodes}')
