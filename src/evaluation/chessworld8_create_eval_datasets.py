import os
import random

import numpy as np
import torch
from tqdm import trange

from envs import make_env
from envs.chessworld import ChessWorld8
from ltl import AvoidSampler, FixedSampler
from ltl.samplers.reach_sampler import ReachSampler
from ltl.samplers.super_sampler import SuperSampler


env_name = 'ChessWorld-v1'
seed = 1
num_episodes = 29
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)


def main():

    path = 'eval_datasets/ChessWorld-v1'
    os.makedirs(f'{path}/worlds', exist_ok=True)

    with open(path + "/tasks.txt", 'r') as file:
        c = 0
        for line in file:
            formula = line.strip()

            sampler = FixedSampler.partial(formula)
            env = make_env(env_name, sampler, render_mode=None)
            all_options = {i: {'init_square': square} for i, square in enumerate(env.FREE_SQUARES)}

            for i in range(num_episodes):
                counter = 29 * c + i
                obs = env.reset(options=all_options[i])
                env.save_world_info(f'{path}/worlds/world_info_{counter}.pkl')
                formula = obs['goal']
                with open(f'{path}/updated_tasks.txt', 'a+') as f:
                    f.write(formula)
                    f.write('\n')
                # ltl2action_formula = obs['ltl2action_goal']
                # with open(f'{path}/tasks_ltl2action.txt', 'a+') as f:
                #     f.write(str(ltl2action_formula))
                #     f.write('\n')
            c += 1


if __name__ == '__main__':
    main()
