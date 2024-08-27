import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from envs import make_env
from ltl import AvoidSampler, FixedSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search import ExhaustiveSearch
from utils.model_store import ModelStore


def main():
    env_name = 'PointLtl2-v0'
    exp = 'eval'
    seed = 2
    deterministic = True
    num_eval_steps = 100
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    sampler = AvoidSampler.partial(2, 1)

    env = make_env(env_name, sampler, render_mode=None, max_steps=1000)
    config = model_configs['default']
    model_store = ModelStore(env_name, exp, seed, None)
    statuses = model_store.load_eval_training_statuses(map_location='cpu')

    xs, sr, vr, steps = [], [], [], []
    pbar = tqdm(statuses)
    for status in pbar:
        model = build_model(env, status, config)
        s, v, num_steps = eval_model(model, env, num_eval_steps, seed, deterministic)
        xs.append(status["num_steps"])
        sr.append(s)
        vr.append(v)
        steps.append(num_steps)
        pbar.set_postfix({'num_steps': status["num_steps"], 'success_rate': s, 'violation_rate': v, 'average_steps': num_steps})
    env.close()
    print(sr)

    df = pd.DataFrame({'num_steps': xs, 'success_rate': sr, 'violation_rate': vr, 'average_steps': steps})
    df.to_csv(f'eval{seed}.csv', index=False)


def eval_model(model, env, num_eval_steps, seed, deterministic):
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    num_successes = 0
    num_violations = 0
    steps = []
    env.reset(seed=seed)
    for _ in range(num_eval_steps):
        agent.reset()
        obs = env.reset()
        info = {'ldba_state_changed': True}
        done = False
        num_steps = 0
        while not done:
            action = agent.get_action(obs, info, deterministic=deterministic)
            action = action.flatten()
            obs, reward, done, info = env.step(action)
            num_steps += 1
            if done:
                if 'success' in info:
                    num_successes += 1
                    steps.append(num_steps)
                elif 'violation' in info:
                    num_violations += 1
    return num_successes / num_eval_steps, num_violations / num_eval_steps, np.mean(steps) if steps else -1

if __name__ == '__main__':
    main()