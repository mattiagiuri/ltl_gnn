import os

import pandas as pd

from config import model_configs
from envs.chessworld import ChessWorld, ChessWorld8
# from evaluation.evaluate_chessworld import evaluate_chessworld_deepsets, evaluate_chessworld_gnn
from generate_formula_assignments import all_simple_reach_avoid
from evaluation.simulate import simulate, simulate_faster, simulate_all
import numpy as np

from model.model import build_model_gnn, build_model
from preprocessing import init_vocab, init_vars
from utils.model_store import ModelStore
from itertools import combinations

sample_vocab = {0: 'PAD', 1: 'EPSILON', 2: 'NULL', 3: 'queen', 4: 'rook', 5: 'knight', 6: 'bishop', 7: 'pawn',
                    8: 'queen&rook', 9: 'queen&bishop', 10: 'queen&pawn&bishop', 11: 'queen&pawn&rook',
                    12: 'knight&rook', 13: 'bishop&rook', 14: 'knight&bishop', 15: 'blank'}

var_names = ['EPSILON', 'NULL', 'queen', 'rook', 'knight', 'bishop', 'pawn', 'blank']
true_vars = ['queen', 'rook', 'knight', 'bishop', 'pawn']

possible_reach = set([sample_vocab[i] for i in range(3, 15)])

env = ChessWorld8()
# vars = set(env.PIECES.keys())

tasks_path = "eval_datasets/ChessWorld-v0/tasks.txt"
env_name = 'ChessWorld-v1'
exp_gnn = "gcn"
exp_gnn_prop = "gcn_prop"
exp_gnn_fine = "gcn_fine"
exp_deepsets = "deepsets_full"
exp_deepsets_prop = "deepsets_prop"
seed = 1
num_episodes = 29
gamma = 0.98
finite = True
render = False
deterministic = True
gnn_gnn = True
gnn_deepsets = False
init_voc = True


def load_models(gnn, exp):
    global init_voc
    global env_name

    if not gnn:
        try:
            config = model_configs[env_name]
        except KeyError:
            config = model_configs["ChessWorld-v1"]
        # exp = "deepsets_full"
    else:
        if init_voc:
            init_vocab(env.get_possible_assignments())
            init_vars(env.get_propositions())

            init_voc = False

        try:
            config = model_configs["gnn_" + env_name]
        except KeyError:
            config = model_configs["gnn_ChessWorld-v1"]
        # exp = "gcn"

    # print(config)
    model_store = ModelStore(env_name, exp, seed, None)
    # model_store.path = "experiments/ppo/ChessWorld-v0/deepsets_full/1"
    # model_store.path = "experiments/ppo/ChessWorld-v0/gcn/1"

    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()

    if gnn:
        model = build_model_gnn(env, training_status, config)
    else:
        model = build_model(env, training_status, config)

    return model


def avoid_x(x):
    all_avoid = combinations(true_vars, x)
    final_formulae = []

    for avoid in all_avoid:
        cur_reach = possible_reach - set(avoid)

        for reach in cur_reach:
            reach_formula = "U(" + reach + ")"
            avoid_formula = "!(" + "|".join(avoid) + ")"

            final_formula = "(" + avoid_formula + reach_formula + ")"
            final_formulae.append(final_formula)

    return final_formulae


def evaluate_chessworld(task, exp_thing, is_gnn, cur_config, seed=seed):
    global init_voc
    successes, avg_steps, adr = simulate_all(env_name, gamma, exp_thing, seed, num_episodes, task, finite,
                                         render, deterministic, is_gnn, cur_config, init_voc=init_voc)

    if init_voc:
        init_voc = False

    return successes


def chessworld8_many_ablation(models_keys, models_names, is_gnns, cur_configs, seed=1, size_range=range(1, 6)):

    # results_dict = {k: {"successes_mean": [], "successes_std": []}
    #                 for k in models_keys}

    results_dict = {k: {"successes_mean": []} for k in models_keys}

    df_index = list(size_range)
    parent = "chessworld8_ablation/stay_update/" + str(seed)
    os.makedirs(parent, exist_ok=True)

    for size in size_range:
        all_formulae = avoid_x(size)

        for key, exp_thing, is_gnn, cur_config in zip(models_keys, models_names, is_gnns, cur_configs):
            cur_successes = []

            # for formula in all_formulae:
            #     cur_successes.append(evaluate_chessworld(formula, exp_thing, is_gnn, cur_config, seed=seed))

            mean = evaluate_chessworld(all_formulae, exp_thing, is_gnn, cur_config, seed=seed)

            # mean = np.mean(cur_successes) / num_episodes
            # std = np.std(cur_successes) / num_episodes

            results_dict[key]["successes_mean"].append(mean)
            # results_dict[key]["successes_std"].append(std)

        df_list = [pd.DataFrame({metric: [val[-1]] for metric, val in cur_db.items()})
                   for _, cur_db in results_dict.items()]
        cur_df_save = pd.concat(df_list, axis=1, keys=models_keys)

        cur_df_name = parent + "/results_" + str(size) + ".csv"

        old_df_save = pd.read_csv(cur_df_name, header=[0, 1], index_col=0)

        cur_df_save = pd.concat([old_df_save, cur_df_save], axis=1)
        cur_df_save.to_csv(cur_df_name)

        print("Done " + str(size))

    final_dfs = [pd.DataFrame(res, index=df_index) for _, res in results_dict.items()]
    results_df = pd.concat(final_dfs, axis=1, keys=models_keys)

    final_name = parent + "/results.csv"
    old_final_df = pd.read_csv(final_name, header=[0, 1], index_col=0)

    results_df = pd.concat([old_final_df, results_df], axis=1)
    results_df.to_csv(final_name)

    return results_df


if __name__ == "__main__":

    # keys_new = ['Deepsets (0.85)', 'Deepsets (0.9)', 'Deepsets (15M)',
    #             'GCN (0.85)', 'GCN (0.9)', 'GCN (15M)',
    #             'GCN (defrosted 25M)']

    # keys_new = ['Deepsets (large avoid)',
    #             'Deepsets (15M)',
    #              'GCN (15M)',
    #             ]

    keys_new = ["Deepsets (formula)"]

    # model_names = [
    #     'deepsets_trial_4',
    #     # 'deepsets_stay_update_4',
    #     # 'deepsets_stay_update_4_fine',
    #     'deepsets_stay_update_4_finest',
    #     # 'gcn_formula_big_skip_6',
    #     # 'gcn_formula_big_skip_6_fine',
    #     'gcn_formula_big_skip_6_finer',
    #     # 'gcn_formula_big_skip_6_finest'
    # ]

    model_names = ["deepsets_update_2"]

    # cur_configs = ["big_sets_ChessWorld-v1"]*2 + ["big_ChessWorld-v1"]*1

    cur_configs = ["big_sets_ChessWorld-v1"]
    # is_gcn = [False]*2 + [True]*1

    is_gcn = [False]

    for seed in range(1, 6):
        chessworld8_many_ablation(keys_new, model_names, is_gcn, cur_configs, seed=seed)
