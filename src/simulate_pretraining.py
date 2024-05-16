import json
from pprint import pprint

import numpy as np
import torch

from envs import make_sequence_env, get_env_attr
from ltl import EventuallySampler, ReachFourSampler
from ltl.samplers.fixed_sampler import FixedSampler
from model.ltl.batched_sequence import BatchedSequence
from model.model import build_model
from model.agent import Agent
from config import model_configs
from utils.model_store import ModelStore


def count_steps() -> int:
    env_name = 'pretraining_PointLtl2-v0'
    exp = 'seq'
    seed = 1

    env = make_sequence_env(env_name)
    config = model_configs['pretraining']
    model_store = ModelStore(env_name, exp, seed, None)
    training_status = model_store.load_training_status(map_location='cpu')
    model = build_model(env, training_status, config, None, False)
    agent = Agent(model)

    obs = env.reset()
    print(obs['goal'])
    done = False
    steps = 0
    while not done:
        action = agent.get_action(obs, deterministic=True).item()
        # action = env.action_space.sample()
        assignment = get_env_attr(env, 'index_to_assignment')[action]
        print([k for k, v in assignment.items() if v])
        obs, reward, done, info = env.step(action)
        steps += 1
        print(reward)

    env.close()
    return steps


def main():
    steps = [count_steps() for _ in range(10)]
    print('Average:', np.mean(steps))


if __name__ == '__main__':
    main()
