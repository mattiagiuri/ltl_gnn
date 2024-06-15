import random
import time

import numpy as np
import pygame
import torch
from tqdm import trange

from envs import make_env
from ltl import EventuallySampler, ReachFourSampler
from ltl.samplers import ReachAvoidSampler
from ltl.samplers.fixed_sampler import FixedSampler
from ltl.samplers.loop_sampler import LoopSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence import RandomSequenceSampler
from sequence.fixed_sequence_sampler import FixedSequenceSampler
from utils.model_store import ModelStore

env_name = 'LetterEnv-v0'
exp = 'q'  # best so far: 64_emb_4_epochs_2_layers
seed = 1
render_modes = [None, 'human', 'path']
render = render_modes[0]

random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

sampler = RandomSequenceSampler.partial(length=2, unique=True)
# sampler = FixedSequenceSampler.partial([('b', 'a')])
deterministic = False
shielding = False

env = make_env(env_name, sampler, ltl=False, max_steps=75, render_mode=render, eval_mode=True)
config = model_configs['letter']
model_store = ModelStore(env_name, exp, seed, None)
training_status = model_store.load_training_status(map_location='cpu')
model = build_model(env, training_status, config)
agent = Agent(model)

num_episodes = 1000

num_successes = 0
num_violations = 0
rets = []
success_mask = []
steps = []

env.reset(seed=seed)

finish = False
pbar = range(num_episodes) if render else trange(num_episodes)
for i in pbar:
    actions = []
    obs = env.reset()
    if render:
        print(obs['goal'])
    done = False
    num_steps = 0
    while not done:
        # TODO: reduce epsilon over training. Using shielding should hopefully help a lot to essentially have a deterministic policy that has a high success rate (does not get stuck in loops -> this is the real problem)
        action = agent.get_action(obs, deterministic=deterministic, shielding=shielding)
        action = action.flatten()[0]
        actions.append(action)
        obs, reward, done, info = env.step(action)
        if render == 'human':
            print(reward)
            finish = env.wait_for_input()
            if finish:
                break
        num_steps += 1
        if done:
            if 'success' in info:
                num_successes += 1
                final_reward = 1
                steps.append(num_steps)  # only count steps for successful episodes
            elif 'violation' in info:
                num_violations += 1
                final_reward = -1
            else:
                final_reward = 0
            rets.append(final_reward * 0.94 ** (num_steps - 1))
            success_mask.append('success' in info)

            if render == 'path':
                print(final_reward)
                env.render_path(actions)
                finish = env.wait_for_input()
            elif not render:
                pbar.set_postfix({'success': num_successes / (i + 1), 'ADR(t)': np.mean(rets),
                                  'ADR(s)': np.mean(np.array(rets)[success_mask]), 'AS': np.mean(steps)})
    if finish:
        break

env.close()
print(f'Success rate: {num_successes / num_episodes}')
print(f'Violation rate: {num_violations / num_episodes}')
print(f'Num total: {num_episodes}')
print(f'ADR (total): {np.mean(rets):.3f}')
print(f'ADR (successful): {np.mean(np.array(rets)[success_mask]):.3f}')
print(f'AS: {np.mean(steps):.3f}')

print(f'{num_successes / num_episodes:.3f},{num_violations / num_episodes:.3f},{np.mean(rets):.3f},{np.mean(np.array(rets)[success_mask]):.3f},{np.mean(steps):.3f}')
