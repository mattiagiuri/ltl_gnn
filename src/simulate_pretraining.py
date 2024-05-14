import functools
import json
import os.path
from pprint import pprint

import numpy as np
import torch

from envs import make_env, get_env_attr
from ltl import EventuallySampler, ReachFourSampler
from ltl.samplers.fixed_sampler import FixedSampler
from model.ltl.batched_sequence import BatchedSequence
from model.model import build_model
from model.agent import Agent
from config import model_configs


def count_steps(formula: str) -> int:
    env_name = 'pretraining_PointLtl2-v0'
    exp = 'rnn'
    seed = 1

    with open('vocab.json', 'r') as f:
        BatchedSequence.VOCAB = json.load(f)

    sampler = FixedSampler.partial_from_formula(formula)
    # sampler = FixedSampler.partial_from_formula('GF blue')
    # sampler = FixedSampler.partial_from_formula('F (red | yellow)')
    env = make_env(env_name, sampler)
    config = model_configs['pretraining']
    training_status = torch.load(f'experiments/ppo/{env_name}/{exp}/{seed}/status.pth', map_location='cpu')
    model = build_model(env, training_status, config, None, False)
    agent = Agent(model)

    obs = env.reset()
    print(obs['goal'])
    done = False
    steps = 0
    while not done:
        action = agent.get_action(obs, deterministic=True).item()
        print(get_env_attr(env, 'index_to_assignment')[action])
        obs, reward, done, info = env.step(action)
        steps += 1
        print(reward)
        # print(obs)

    env.close()
    return steps


def main():
    propositions = ['green', 'blue', 'yellow', 'magenta']
    # formulas = [f'F ({a} & (F {b}))' for a in propositions for b in propositions if a != b]\
    formulas = ReachFourSampler(propositions).tasks
    steps = {formula: count_steps(formula) for formula in formulas}
    pprint(steps)
    print('Average:', np.mean(list(steps.values())))


if __name__ == '__main__':
    main()
