import random

import numpy as np
import torch
from tqdm import tqdm
import pickle

from envs import make_env
from ltl import FixedSampler
from ltl.automata import LDBASequence
from model.model import build_model, build_model_gnn
from model.agent import Agent
from config import model_configs
from sequence.search import ExhaustiveSearch
from utils.model_store import ModelStore
from preprocessing.vocab import init_vocab, init_vars, augment_vars
import argparse
from envs.chessworld import ChessWorld
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=['PointLtl2-v0', 'LetterEnv-v0', 'FlatWorld-v0', 'ChessWorld-v1'], default='ChessWorld-v1')
    parser.add_argument('--exp', type=str, default='gcn_formula_update')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_episodes', type=int, default=6)
    parser.add_argument('--formula', type=str, default='(F (pawn & F (rook)))')  # (!(knight | pawn) U queen)
    parser.add_argument('--finite', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--render', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--gnn', action=argparse.BooleanOptionalAction, default=True)
    # parser.add_argument('--from_seq', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    gamma = 0.94 if args.env == 'LetterEnv-v0' else 0.998 if args.env == 'PointLtl2-v0' else 0.98

    return simulate(args.env, gamma, args.exp, args.seed, args.num_episodes, args.formula, args.finite, args.render, args.deterministic, args.gnn, init_voc=True)


def simulate_all(env, gamma, exp, seed, num_episodes, formulae, finite, render, deterministic, gnn, cur_config, init_voc=False):
    env_name = env
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    sampler = FixedSampler.partial(formulae[0])
    env = make_env(env_name, sampler, render_mode='human' if render else None)
    all_options = {i: {'init_square': square} for i, square in enumerate(env.FREE_SQUARES)}

    if init_voc:
        init_vocab(env.get_possible_assignments())
        init_vars(env.get_propositions())
        augment_vars()


    # if not gnn:
    #     try:
    #         config = model_configs[env_name]
    #     except KeyError:
    #         config = model_configs["ChessWorld-v1"]
    #     # exp = "deepsets_full"
    # else:
    #
    #
    #     try:
    #         config = model_configs["gnn_" + env_name]
    #     except KeyError:
    #         config = model_configs["gnn_ChessWorld-v1"]

    config = model_configs[cur_config]
        # exp = "gcn"

    # print(config)
    model_store = ModelStore(env_name, exp, seed, None)
    # model_store.path = "experiments/ppo/ChessWorld-v0/deepsets_full/1"
    # model_store.path = "experiments/ppo/ChessWorld-v0/gcn/1"

    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()

    model = build_model(env, training_status, config)

    all_successes = []
    all_steps = []
    all_adr = []

    for formula in formulae:
        # print(formula)

        sampler = FixedSampler.partial(formula)
        env = make_env(env_name, sampler, render_mode='human' if render else None)

        # if init_voc:
        #     init_vocab(env.get_possible_assignments())
        #     init_vars(env.get_propositions())
        #     augment_vars()

        props = set(env.get_propositions())
        search = ExhaustiveSearch(model, props, num_loops=2)

        # print(search.all_sequences())
        # print(search)
        # return
        agent = Agent(model, search=search, propositions=props, verbose=render)

        num_successes = 0
        num_violations = 0
        num_accepting_visits = 0
        steps = []
        rets = []

        env.reset(seed=seed)

        pbar = range(num_episodes)
        # if not render:
        #     pbar = tqdm(pbar)
        for i in pbar:
            obs, info = env.reset(options=all_options[i]), {}
            # obs, info = env.initial_square_reset(), {}

            if render:
                print(obs['goal'])
            agent.reset()
            done = False
            num_steps = 0
            while not done:
                action = agent.get_action(obs, info, deterministic=deterministic)
                action = action.flatten()
                if action.shape == (1,):
                    action = action[0]
                obs, reward, done, info = env.step(action)
                num_steps += 1
                if done:
                    # print(num_steps)
                    # print(env.agent_pos)
                    if finite:
                        final_reward = int('success' in info)
                        if 'success' in info:
                            num_successes += 1
                            steps.append(num_steps)
                        elif 'violation' in info:
                            num_violations += 1
                        rets.append(final_reward * gamma ** (num_steps - 1))
                        # if not render:
                        #     pbar.set_postfix({
                        #         'S': num_successes / (i + 1),
                        #         'V': num_violations / (i + 1),
                        #         'ADR': np.mean(rets),
                        #         'AS': np.mean(steps),
                        #     })
                    else:
                        num_accepting_visits += info['num_accepting_visits']
                        if not render:
                            pbar.set_postfix({
                                'A': num_accepting_visits / (i + 1),
                            })

        env.close()
        if finite:
            success_rate = num_successes / num_episodes
            violation_rate = num_violations / num_episodes
            average_steps = np.mean(steps)
            adr = np.mean(rets)
            # print(f'{seed}: {success_rate:.3f},{violation_rate:.3f},{adr:.3f},{average_steps:.3f}')
            all_successes.append(num_successes)
            all_steps.append(average_steps)
            all_adr.append(adr)

        else:
            average_visits = num_accepting_visits / num_episodes
            print(f'{seed}: {average_visits:.3f}')
            return average_visits

    # num_episodes = len(all_options) * len(formulae)
    # assert(len(all_options) == 29)
    return np.mean(all_successes) / len(all_options), np.mean(all_steps) / len(all_options), np.mean(all_adr) / len(all_options)


def simulate(env, gamma, exp, seed, num_episodes, formula, finite, render, deterministic, gnn, cur_config, init_voc=False, chess=False):
    print(formula)
    env_name = env
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    sampler = FixedSampler.partial(formula)
    env = make_env(env_name, sampler, render_mode='human' if render else None)

    if chess:
        all_options = {i: {'init_square': square} for i, square in enumerate(env.FREE_SQUARES)}

    if init_voc:
        init_vocab(env.get_possible_assignments())
        init_vars(env.get_propositions())
        augment_vars()

    # if not gnn:
    #     try:
    #         config = model_configs[env_name]
    #     except KeyError:
    #         config = model_configs["ChessWorld-v1"]
    #     # exp = "deepsets_full"
    # else:
    #
    #
    #     try:
    #         config = model_configs["gnn_" + env_name]
    #     except KeyError:
    #         config = model_configs["gnn_ChessWorld-v1"]

    config = model_configs[cur_config]
        # exp = "gcn"

    # print(config)
    model_store = ModelStore(env_name, exp, seed, None)
    # model_store.path = "experiments/ppo/ChessWorld-v0/deepsets_full/1"
    # model_store.path = "experiments/ppo/ChessWorld-v0/gcn/1"

    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()

    model = build_model(env, training_status, config)

    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)

    # print(search.all_sequences())
    # print(search)
    # return
    agent = Agent(model, search=search, propositions=props, verbose=render)

    check_accepting_cycle = deterministic and not finite

    num_successes = 0
    num_violations = 0
    num_accepting_visits = 0
    steps = []
    rets = []

    env.reset(seed=seed)

    pbar = range(num_episodes)
    if not render:
        pbar = tqdm(pbar)
    for i in pbar:
        if chess:
            obs, info = env.reset(options=all_options[i]), {}
        else:
            obs, info = env.reset(), {}

        state_to_step = dict()
        accepting_times = set()
        t = 0
        state_to_step[(tuple(obs['features']), obs['ldba_state'])] = t
        if render:
            print(obs['goal'])
        agent.reset()
        done = False
        num_steps = 0
        while not done:
            action = agent.get_action(obs, info, deterministic=deterministic)
            action = action.flatten()
            if action.shape == (1,):
                action = action[0]
            obs, reward, done, info = env.step(action)
            if info['accepting']:
                accepting_times.add(t)
            t += 1
            if check_accepting_cycle:
                if (tuple(obs['features']), obs['ldba_state']) in state_to_step:  # loop
                    state_time = state_to_step[(tuple(obs['features']), obs['ldba_state'])]
                    if any(accepting_time >= state_time for accepting_time in accepting_times):
                        num_successes += 1
                    break

            state_to_step[(tuple(obs['features']), obs['ldba_state'])] = t
            num_steps += 1
            if done:
                # print(num_steps)
                # print(env.agent_pos)
                if finite:
                    final_reward = int('success' in info)
                    if 'success' in info:
                        num_successes += 1
                        steps.append(num_steps)
                    elif 'violation' in info:
                        num_violations += 1
                    rets.append(final_reward * gamma ** (num_steps - 1))
                    if not render:
                        pbar.set_postfix({
                            'S': num_successes / (i + 1),
                            'V': num_violations / (i + 1),
                            'ADR': np.mean(rets),
                            'AS': np.mean(steps),
                        })
                else:
                    num_accepting_visits += info['num_accepting_visits']
                    if not render:
                        pbar.set_postfix({
                            'A': num_accepting_visits / (i + 1),
                        })

    env.close()
    if finite:
        success_rate = num_successes / num_episodes
        violation_rate = num_violations / num_episodes
        average_steps = np.mean(steps)
        adr = np.mean(rets)
        # print(f'{seed}: {success_rate:.3f},{violation_rate:.3f},{adr:.3f},{average_steps:.3f}')
        return num_successes, average_steps, adr
    else:
        # average_visits = num_accepting_visits / num_episodes
        # print(f'{seed}: {average_visits:.3f}')
        # return average_visits
        return num_successes / num_episodes


def simulate_faster(env, gamma, exp, seed, num_episodes, formula, finite, render, deterministic, model, init_voc=False):
    env_name = env
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    sampler = FixedSampler.partial(formula)
    env = make_env(env_name, sampler, render_mode='human' if render else None)
    all_options = {i: {'init_square': square} for i, square in enumerate(env.FREE_SQUARES)}

    if init_voc:
        init_vocab(env.get_possible_assignments())
        init_vars(env.get_propositions())

    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)

    # print(search.all_sequences())
    # print(search)
    # return
    agent = Agent(model, search=search, propositions=props, verbose=render)

    num_successes = 0
    num_violations = 0
    num_accepting_visits = 0
    steps = []
    rets = []

    env.reset(seed=seed)

    pbar = range(num_episodes)
    # if not render:
    #     pbar = tqdm(pbar)
    for i in pbar:
        obs, info = env.reset(options=all_options[i]), {}
        # obs, info = env.initial_square_reset(), {}

        if render:
            print(obs['goal'])
        agent.reset()
        done = False
        num_steps = 0
        while not done:
            action = agent.get_action(obs, info, deterministic=deterministic)
            action = action.flatten()
            if action.shape == (1,):
                action = action[0]
            obs, reward, done, info = env.step(action)
            num_steps += 1
            if done:
                # print(num_steps)
                # print(env.agent_pos)
                if finite:
                    final_reward = int('success' in info)
                    if 'success' in info:
                        num_successes += 1
                        steps.append(num_steps)
                    elif 'violation' in info:
                        num_violations += 1
                    rets.append(final_reward * gamma ** (num_steps - 1))
                    if not render:
                        pbar.set_postfix({
                            'S': num_successes / (i + 1),
                            'V': num_violations / (i + 1),
                            'ADR': np.mean(rets),
                            'AS': np.mean(steps),
                        })
                else:
                    num_accepting_visits += info['num_accepting_visits']
                    if not render:
                        pbar.set_postfix({
                            'A': num_accepting_visits / (i + 1),
                        })

    env.close()
    if finite:
        success_rate = num_successes / num_episodes
        violation_rate = num_violations / num_episodes
        average_steps = np.nanmean(steps)
        adr = np.nanmean(rets)
        # print(f'{seed}: {success_rate:.3f},{violation_rate:.3f},{adr:.3f},{average_steps:.3f}')
        return num_successes, average_steps, adr
    else:
        average_visits = num_accepting_visits / num_episodes
        print(f'{seed}: {average_visits:.3f}')
        return average_visits


def simulate_flat(env, gamma, exp, seed, num_episodes, formula, finite, render, deterministic, gnn, cur_config, init_voc=False, chess=False):
    print(formula)
    env_name = env
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    sampler = FixedSampler.partial(formula)
    env = make_env(env_name, sampler, render_mode='human' if render else None)

    if chess:
        all_options = {i: {'init_square': square} for i, square in enumerate(env.FREE_SQUARES)}

    if init_voc:
        init_vocab(env.get_possible_assignments())
        init_vars(env.get_propositions())
        augment_vars()

    # if not gnn:
    #     try:
    #         config = model_configs[env_name]
    #     except KeyError:
    #         config = model_configs["ChessWorld-v1"]
    #     # exp = "deepsets_full"
    # else:
    #
    #
    #     try:
    #         config = model_configs["gnn_" + env_name]
    #     except KeyError:
    #         config = model_configs["gnn_ChessWorld-v1"]

    config = model_configs[cur_config]
        # exp = "gcn"

    # print(config)
    model_store = ModelStore(env_name, exp, seed, None)
    # model_store.path = "experiments/ppo/ChessWorld-v0/deepsets_full/1"
    # model_store.path = "experiments/ppo/ChessWorld-v0/gcn/1"

    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()

    model = build_model(env, training_status, config)

    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)

    # print(search.all_sequences())
    # print(search)
    # return
    agent = Agent(model, search=search, propositions=props, verbose=render)

    num_successes = 0
    num_violations = 0
    num_accepting_visits = 0
    steps = []
    rets = []

    env.reset(seed=seed)

    pbar = range(num_episodes)
    if not render:
        pbar = tqdm(pbar)
    for i in pbar:
        # print(os.getcwd())
        with open(f'eval_datasets/FlatWorld-v0/worlds3/world_info_{i}.pkl', 'rb') as file:
            world_info = pickle.load(file)
        if chess:
            obs, info = env.reset(options=all_options[i]), {}
        else:
            obs, info = env.reset(options={'world_info': world_info}), {}

        if render:
            print(obs['goal'])
        agent.reset()
        done = False
        num_steps = 0
        while not done:
            action = agent.get_action(obs, info, deterministic=deterministic)
            action = action.flatten()
            if action.shape == (1,):
                action = action[0]
            obs, reward, done, info = env.step(action)
            num_steps += 1
            if done:
                # print(num_steps)
                # print(env.agent_pos)
                if finite:
                    final_reward = int('success' in info)
                    if 'success' in info:
                        num_successes += 1
                        steps.append(num_steps)
                    elif 'violation' in info:
                        num_violations += 1
                    rets.append(final_reward * gamma ** (num_steps - 1))
                    if not render:
                        pbar.set_postfix({
                            'S': num_successes / (i + 1),
                            'V': num_violations / (i + 1),
                            'ADR': np.mean(rets),
                            'AS': np.mean(steps),
                        })
                else:
                    num_accepting_visits += info['num_accepting_visits']
                    if not render:
                        pbar.set_postfix({
                            'A': num_accepting_visits / (i + 1),
                        })

    env.close()
    if finite:
        success_rate = num_successes / num_episodes
        violation_rate = num_violations / num_episodes
        average_steps = np.mean(steps)
        adr = np.mean(rets)
        # print(f'{seed}: {success_rate:.3f},{violation_rate:.3f},{adr:.3f},{average_steps:.3f}')
        return num_successes, average_steps, adr
    else:
        average_visits = num_accepting_visits / num_episodes
        print(f'{seed}: {average_visits:.3f}')
        return average_visits
if __name__ == '__main__':
    # os.chdir("..")
    # os.chdir("..")
    main()
