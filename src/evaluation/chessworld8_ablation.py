import os

import pandas as pd

from config import model_configs
from envs.chessworld import ChessWorld, ChessWorld8
# from evaluation.evaluate_chessworld import evaluate_chessworld_deepsets, evaluate_chessworld_gnn
from generate_formula_assignments import all_simple_reach_avoid
from evaluation.simulate import simulate, simulate_faster
import numpy as np

from model.model import build_model_gnn, build_model
from preprocessing import init_vocab, init_vars
from utils.model_store import ModelStore

env = ChessWorld8()
vars = set(env.PIECES.keys())

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


def evaluate_chessworld_deepsets(task, exp_deepsets=exp_deepsets, fast=False, model=None):
    if fast:
        successes, avg_steps, adr = simulate_faster(env_name, gamma, exp_deepsets, seed, num_episodes, task, finite,
                                             render, deterministic, model, init_voc=False)

        return successes

    global init_voc
    successes, avg_steps, adr = simulate(env_name, gamma, exp_deepsets, seed, num_episodes, task, finite,
                                         render, deterministic, gnn_deepsets, init_voc=init_voc)

    if init_voc:
        init_voc=False
    return successes


def evaluate_chessworld_gcn(task, exp_gnn=exp_gnn, fast=False, model=None):
    if fast:
        successes, avg_steps, adr = simulate(env_name, gamma, exp_gnn, seed, num_episodes, task, finite,
                                             render, deterministic, model, init_voc=False)

        return successes

    global init_voc
    successes, avg_steps, adr = simulate(env_name, gamma, exp_gnn, seed, num_episodes, task, finite,
                                         render, deterministic, gnn_gnn, init_voc=init_voc)

    if init_voc:
        init_voc = False

    return successes


def evaluate_chessworld_gcn_prop(task, exp_gnn_prop=exp_gnn_prop, fast=False, model=None):
    if fast:
        successes, avg_steps, adr = simulate(env_name, gamma, exp_gnn_prop, seed, num_episodes, task, finite,
                                             render, deterministic, model, init_voc=False)

        return successes

    global init_voc
    successes, avg_steps, adr = simulate(env_name, gamma, exp_gnn_prop, seed, num_episodes, task, finite,
                                         render, deterministic, gnn_gnn, init_voc=init_voc)

    if init_voc:
        init_voc = False

    return successes


def multiset_size_ablation(goal, size_range=range(3, 10)):
    results_gcn = {"successes_mean": [], "successes_std": []}
    results_deepsets = {"successes_mean": [], "successes_std": []}
    results_deepsets_prop = {"successes_mean": [], "successes_std": []}

    df_index = list(size_range)

    for size in size_range:
        all_formulae = all_simple_reach_avoid(env, vars, goal, size)
        successes_gcn = []
        successes_deepsets = []
        successes_deepsets_prop = []

        for formula in all_formulae:
            successes_gcn.append(evaluate_chessworld_gcn(formula))
            successes_deepsets.append(evaluate_chessworld_deepsets(formula))
            successes_deepsets_prop.append(evaluate_chessworld_deepsets(formula, exp_deepsets=exp_deepsets_prop))

        for cur_results, database in zip([successes_gcn, successes_deepsets, successes_deepsets_prop],
                                         [results_gcn, results_deepsets, results_deepsets_prop]):
            mean, std = np.mean(cur_results) / num_episodes, np.std(cur_results) / (num_episodes ** 2)
            database["successes_mean"].append(mean)
            database["successes_std"].append(std)

        os.makedirs("chessworld8_ablation/" + goal, exist_ok=True)

        cur_df_save = pd.concat([pd.DataFrame({i: [j[-1]] for i, j in cur_db.items()})
                                        for cur_db in [results_gcn, results_deepsets, results_deepsets_prop]],
                            axis=1, keys=["GCN", "Deepsets", "Deepsets (prop training)"])

        cur_df_save.to_csv("chessworld8_ablation/" + goal + "/results_" + str(size) + ".csv")

        print("Done " + str(size))


    gcn = pd.DataFrame(results_gcn, index=df_index)
    deepsets = pd.DataFrame(results_deepsets, index=df_index)
    deepsets_prop = pd.DataFrame(results_deepsets_prop, index=df_index)

    results_df = pd.concat([gcn, deepsets, deepsets_prop], axis=1, keys=["GCN", "Deepsets", "Deepsets (prop training)"])

    results_df.to_csv("chessworld8_ablation/" + goal + "/results.csv")

    return results_df


def add_multiset_size_ablation(goal, eval_func=evaluate_chessworld_gcn_prop, size_range=range(3, 10), name="GCN (prop training)"):
    results = {"successes_mean": [], "successes_std": []}

    for size in size_range:
        all_formulae = all_simple_reach_avoid(env, vars, goal, size)
        successes = []

        for formula in all_formulae:
            successes.append(eval_func(formula))

        mean, std = np.mean(successes) / num_episodes, np.std(successes) / (num_episodes ** 2)
        results["successes_mean"].append(mean)
        results["successes_std"].append(std)

        os.makedirs("chessworld8_ablation/" + goal, exist_ok=True)

        cur_data = {i: [j[-1]] for i, j in results.items()}
        path = "chessworld8_ablation/" + goal + "/results_" + str(size) + ".csv"

        _ = augment_results(path, cur_data, name=name)

        print("Done " + str(size))

    path = "chessworld8_ablation/" + goal + "/results.csv"
    results_df = augment_results(path, results, name=name)

    return results_df


def fast_multiset_size_ablation(goal, size_range=range(3, 10)):
    results_gcn = {"successes_mean": [], "successes_std": []}
    results_deepsets = {"successes_mean": [], "successes_std": []}
    results_deepsets_prop = {"successes_mean": [], "successes_std": []}

    df_index = list(size_range)

    gcn = load_models(gnn=gnn_gnn, exp=exp_gnn)
    deepsets = load_models(gnn=gnn_deepsets, exp=exp_deepsets)
    deepsets_prop = load_models(gnn=gnn_deepsets, exp=exp_deepsets_prop)

    for size in size_range:
        all_formulae = all_simple_reach_avoid(env, vars, goal, size)
        successes_gcn = []
        successes_deepsets = []
        successes_deepsets_prop = []

        for formula in all_formulae:
            successes_gcn.append(evaluate_chessworld_gcn(formula, fast=True, model=gcn))
            successes_deepsets.append(evaluate_chessworld_deepsets(formula, fast=True, model=deepsets))
            successes_deepsets_prop.append(evaluate_chessworld_deepsets(formula, exp_deepsets=exp_deepsets_prop, fast=True, model=deepsets_prop))

        for cur_results, database in zip([successes_gcn, successes_deepsets, successes_deepsets_prop],
                                         [results_gcn, results_deepsets, results_deepsets_prop]):
            mean, std = np.mean(cur_results) / num_episodes, np.std(cur_results) / (num_episodes ** 2)
            database["successes_mean"].append(mean)
            database["successes_std"].append(std)

        os.makedirs("chessworld8_ablation/" + goal, exist_ok=True)

        cur_df_save = pd.concat([pd.DataFrame({i: [j[-1]] for i, j in cur_db.items()})
                                        for cur_db in [results_gcn, results_deepsets, results_deepsets_prop]],
                            axis=1, keys=["GCN", "Deepsets", "Deepsets (prop training)"])

        cur_df_save.to_csv("chessworld8_ablation/" + goal + "/results_" + str(size) + ".csv")

        print("Done " + str(size))

    gcn_df = pd.DataFrame(results_gcn, index=df_index)
    deepsets_df = pd.DataFrame(results_deepsets, index=df_index)
    deepsets_prop_df = pd.DataFrame(results_deepsets_prop, index=df_index)

    results_df = pd.concat([gcn_df, deepsets_df, deepsets_prop_df], axis=1, keys=["GCN", "Deepsets", "Deepsets (prop training)"])

    results_df.to_csv("chessworld8_ablation/" + goal + "/results.csv")

    return results_df


def augment_results(path, additional_data, name="GCN (prop training)"):
    df = pd.read_csv(path, index_col=0, header=[0, 1])

    # Simulate additional data (replace this with actual data)
    # additional_data = {
    #     "successes_mean": [0.85, 0.78, 0.72, 0.65, 0.58, 0.50, 0.42],  # Example means
    #     "successes_std": [0.005, 0.006, 0.007, 0.008, 0.009, 0.009, 0.008],  # Example stds
    # }
    #
    # additional_data = {
    #     "successes_mean": [0.85],  # Example means
    #     "successes_std": [0.005],  # Example stds
    # }
    index = df.index

    new_model_df = pd.DataFrame(additional_data, index=index)
    new_model_df.columns = pd.MultiIndex.from_product([[name], new_model_df.columns])

    # Concatenate with the original DataFrame
    df = pd.concat([df, new_model_df], axis=1)

    df.to_csv(path)

    return df


if __name__ == "__main__":
    # goal = "(bishop & rook)"
    # goal = "(queen & bishop)"

    # goal1 = "(pawn)"
    # goal2 = "(knight & rook)"

    # goal3 = "(queen & !pawn)"
    # results = multiset_size_ablation(goal=goal1, size_range=range(3, 10))
    # print(results)

    # results1 = multiset_size_ablation(goal=goal3, size_range=range(3, 10))
    # print(results1)

    goals = ["(knight)", "(pawn)", "(bishop & rook)", "(queen & bishop)", "(knight & rook)"]
    for goal in goals[3:]:
        print(goal)
        add_multiset_size_ablation(goal)

    # print(augment_results("chessworld8_ablation/(knight)/results_9.csv", []))

