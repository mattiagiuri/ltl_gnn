from evaluation.simulate import simulate, simulate_flat
from envs.chessworld import ChessWorld
import pandas as pd
import re
from preprocessing.vocab import init_vocab, init_vars
import os
import numpy as np

tasks_path = "eval_datasets/FlatWorld-v0/eval_tasks.txt"
env = 'FlatWorld-v0'
exp_gnn = "gcn"
exp_gnn_prop = "gcn_prop"
exp_gnn_fine = "gcn_fine"
exp_deepsets = "deepsets_full"
exp_deepsets_prop = "deepsets_prop"
seed = 1
num_episodes = 50
gamma = 0.98
finite = True
render = False
deterministic = True
gnn_gnn = True
gnn_deepsets=False
init_voc = True

os.makedirs("results_flatworld/FlatWorld-v0", exist_ok=True)


def read_tasks(tasks_path=tasks_path):
    task_dict = {}

    with open(tasks_path, 'r') as file:
        cur_set = None
        for line in file:
            if line.startswith("Avoid") or line.startswith("Reach"):
                task_dict[line[:-1]] = []
                cur_set = line[:-1]
            else:
                name = line.split()[0]
                task = re.search(r'\((.*)\)', line).group(0)
                task_dict[cur_set].append((name, task))

    print(task_dict)
    return task_dict


def evaluate_gnn(cur_config, save=False, exp_gnn=exp_gnn, tasks=tasks_path, seed=seed):
    global init_voc
    task_list = read_tasks(tasks)
    results = {"Task Set": [], "Task ID": [], "Successes": [], "Avg Steps": [], "Avg Discounted Return": []}

    for task_set, tasks in task_list.items():
        for name, task in tasks:
            successes, avg_steps, adr = simulate_flat(env, gamma, exp_gnn, seed, num_episodes, task, finite,
                                                 render, deterministic, gnn_gnn, cur_config, init_voc)
            if init_voc:
                init_voc = False

            results["Task Set"].append(task_set)
            results["Task ID"].append(name)
            results["Successes"].append(successes)
            results["Avg Steps"].append(avg_steps)
            results["Avg Discounted Return"].append(adr)

    df_results = pd.DataFrame(results)
    df_results.set_index(["Task Set", "Task ID"], inplace=True)

    if save:
        dir_name = "results_flatworld/FlatWorld-v0/" + str(seed)
        os.makedirs(dir_name, exist_ok=True)

        df_results.to_csv(dir_name + "/" + env + "_" + exp_gnn + "2.csv")

    return df_results


def evaluate_gnn_infinite(cur_config, save=False, exp_gnn=exp_gnn, seed=seed):
    global init_voc
    task_list = read_tasks("eval_datasets/FlatWorld-v0/infinite_tasks.txt")
    results = {"Task Set": [], "Task ID": [], "Accepting visits": []}

    for task_set, tasks in task_list.items():
        for name, task in tasks:
            acc_visits = simulate_flat(env, gamma, exp_gnn, seed, num_episodes, task, not finite,
                                                 render, not deterministic, gnn_gnn, cur_config, init_voc)
            if init_voc:
                init_voc = False

            results["Task Set"].append(task_set)
            results["Task ID"].append(name)
            results["Accepting visits"].append(acc_visits)


    df_results = pd.DataFrame(results)
    df_results.set_index(["Task Set", "Task ID"], inplace=True)

    if save:
        dir_name = "results_flatworld/FlatWorld-v0/" + str(seed)
        os.makedirs(dir_name, exist_ok=True)

        df_results.to_csv(dir_name + "/" + env + "_" + exp_gnn + "_inf2.csv")

    return df_results


def evaluate_deepsets(cur_config, save=False, exp_deepsets=exp_deepsets, tasks=tasks_path, seed=seed):
    task_list = read_tasks(tasks)
    results = {"Task Set": [], "Task ID": [], "Successes": [], "Avg Steps": [], "Avg Discounted Return": []}

    for task_set, tasks in task_list.items():
        for name, task in tasks:
            successes, avg_steps, adr = simulate_flat(env, gamma, exp_deepsets, seed, num_episodes, task, finite,
                                                 render, deterministic, gnn_deepsets, cur_config)

            results["Task Set"].append(task_set)
            results["Task ID"].append(name)
            results["Successes"].append(successes)
            results["Avg Steps"].append(avg_steps)
            results["Avg Discounted Return"].append(adr)

    df_results = pd.DataFrame(results)
    df_results.set_index(["Task Set", "Task ID"], inplace=True)

    if save:
        dir_name = "results_flatworld/FlatWorld-v0/" + str(seed)
        os.makedirs(dir_name, exist_ok=True)

        df_results.to_csv(dir_name + "/" + env + "_" + exp_deepsets + "2.csv")

    return df_results


def evaluate_deepsets_infinite(cur_config, save=False, exp_deepsets=exp_deepsets, seed=seed):
    task_list = read_tasks("eval_datasets/FlatWorld-v0/infinite_tasks.txt")
    results = {"Task Set": [], "Task ID": [], "Accepting visits": []}

    for task_set, tasks in task_list.items():
        for name, task in tasks:
            acc_visits = simulate_flat(env, gamma, exp_deepsets, seed, num_episodes, task, not finite,
                                                 render, not deterministic, gnn_deepsets, cur_config)

            results["Task Set"].append(task_set)
            results["Task ID"].append(name)
            results["Accepting visits"].append(acc_visits)

    df_results = pd.DataFrame(results)
    df_results.set_index(["Task Set", "Task ID"], inplace=True)

    if save:
        dir_name = "results_flatworld/FlatWorld-v0/" + str(seed)
        os.makedirs(dir_name, exist_ok=True)

        df_results.to_csv(dir_name + "/" + env + "_" + exp_deepsets + "_inf2.csv")

    return df_results


def evaluate_chessworld8_gnn_nondet(save=False, exp_gnn=exp_gnn):
    global init_voc
    task_list = read_tasks()
    results = {"Task Set": [], "Task ID": [], "Avg Successes x/29": [], "Std Successes": [], "Avg Steps": [], "Avg Discounted Return": []}

    for task_set, tasks in task_list.items():
        for name, task in tasks:
            cur_successes = []
            cur_avg_steps = []
            cur_adr = []

            for _ in range(1):
                successes, avg_steps, adr = simulate(env, gamma, exp_gnn, seed, num_episodes, task, finite,
                                                     render, not deterministic, gnn_gnn, init_voc)

                cur_successes.append(successes)
                cur_avg_steps.append(avg_steps)
                cur_adr.append(adr)

                if init_voc:
                    init_voc = False

            results["Task Set"].append(task_set)
            results["Task ID"].append(name)
            results["Avg Successes x/29"].append(np.mean(cur_successes))
            results["Std Successes"].append(np.std(cur_successes))
            results["Avg Steps"].append(np.mean(cur_avg_steps))
            results["Avg Discounted Return"].append(np.mean(cur_adr))

    df_results = pd.DataFrame(results)
    df_results.set_index(["Task Set", "Task ID"], inplace=True)

    if save:
        df_results.to_csv("results_chessworld/ChessWorld-v1/non_det_" + env + "_" + exp_gnn + ".csv")

    return df_results


def evaluate_chessworld8_deepsets_nondet(save=False, exp_deepsets=exp_deepsets):
    global init_voc
    task_list = read_tasks()
    results = {"Task Set": [], "Task ID": [], "Avg Successes x/29": [], "Std Successes": [], "Avg Steps": [],
               "Avg Discounted Return": []}

    for task_set, tasks in task_list.items():
        for name, task in tasks:
            cur_successes = []
            cur_avg_steps = []
            cur_adr = []

            for _ in range(1):
                successes, avg_steps, adr = simulate(env, gamma, exp_deepsets, seed, num_episodes, task, finite,
                                                     render, not deterministic, gnn_deepsets, init_voc)

                cur_successes.append(successes)
                cur_avg_steps.append(avg_steps)
                cur_adr.append(adr)

                if init_voc:
                    init_voc = False

            results["Task Set"].append(task_set)
            results["Task ID"].append(name)
            results["Avg Successes x/29"].append(np.mean(cur_successes))
            results["Std Successes"].append(np.std(cur_successes))
            results["Avg Steps"].append(np.mean(cur_avg_steps))
            results["Avg Discounted Return"].append(np.mean(cur_adr))

    df_results = pd.DataFrame(results)
    df_results.set_index(["Task Set", "Task ID"], inplace=True)

    if save:
        df_results.to_csv("results_chessworld/ChessWorld-v1/non_det_" + env + "_" + exp_deepsets + ".csv")

    return df_results

if __name__ == "__main__":

    # evaluate_chessworld8_deepsets("a", True, "deepsets_stay")
    # evaluate_chessworld8_gnn('big_ChessWorld-v1', True,'gcn_formula_big_skip_6')
    # evaluate_chessworld8_deepsets(True, exp_deepsets_prop)
    # evaluate_chessworld8_gnn(True, exp_gnn)
    # evaluate_chessworld8_deepsets_nondet(True)
    # evaluate_chessworld8_deepsets_nondet(True, exp_deepsets_prop)
    # evaluate_chessworld8_gnn_nondet(True)

    # evaluate_chessworld8_gnn('big_ChessWorld-v1', True, 'gcn_formula_big_skip_6', tasks='eval_datasets/ChessWorld-v1/finite_tasks.txt')
    # evaluate_chessworld8_gnn_infinite('big_ChessWorld-v1', True, 'gcn_formula_big_skip_6')
    #
    # evaluate_chessworld8_gnn('big_ChessWorld-v1', True, 'gcn_formula_big_skip_6_fine', tasks='eval_datasets/ChessWorld-v1/finite_tasks.txt')
    # evaluate_chessworld8_gnn_infinite('big_ChessWorld-v1', True, 'gcn_formula_big_skip_6_fine')
    #
    # evaluate_chessworld8_gnn('big_ChessWorld-v1', True, 'gcn_formula_big_skip_6_finer', tasks='eval_datasets/ChessWorld-v1/finite_tasks.txt')
    # evaluate_chessworld8_gnn_infinite('big_ChessWorld-v1', True, 'gcn_formula_big_skip_6_finer')
    #
    # evaluate_chessworld8_gnn('big_ChessWorld-v1', True, 'gcn_formula_big_skip_6_finest', tasks='eval_datasets/ChessWorld-v1/finite_tasks.txt')
    # evaluate_chessworld8_gnn_infinite('big_ChessWorld-v1', True, 'gcn_formula_big_skip_6_finest')
    # #
    # evaluate_chessworld8_deepsets('big_sets_ChessWorld-v1', True, 'deepsets_stay_update_4', tasks='eval_datasets/ChessWorld-v1/finite_tasks.txt')
    # evaluate_chessworld8_deepsets_infinite('big_sets_ChessWorld-v1', True, 'deepsets_stay_update_4')
    #
    # evaluate_chessworld8_deepsets('big_sets_ChessWorld-v1', True, 'deepsets_stay_update_4_fine',
    #                               tasks='eval_datasets/ChessWorld-v1/finite_tasks.txt')
    # evaluate_chessworld8_deepsets_infinite('big_sets_ChessWorld-v1', True, 'deepsets_stay_update_4_fine')
    #
    # evaluate_chessworld8_deepsets('big_sets_ChessWorld-v1', True, 'deepsets_stay_update_4_finest',
    #                               tasks='eval_datasets/ChessWorld-v1/finite_tasks.txt')
    # evaluate_chessworld8_deepsets_infinite('big_sets_ChessWorld-v1', True, 'deepsets_stay_update_4_finest')

    for cur_seed in range(1, 6):
        print(cur_seed)
        print("deepsets")
        # evaluate_chessworld8_deepsets('big_sets_ChessWorld-v1', True, 'deepsets_stay_update_4_finest',
        #                               tasks='eval_datasets/ChessWorld-v1/finite_tasks.txt', seed=cur_seed)
        # evaluate_chessworld8_deepsets_infinite('big_sets_ChessWorld-v1', True, 'deepsets_stay_update_4_finest', seed=cur_seed)

        evaluate_deepsets('FlatWorld-v0', True, 'deepsets_stay',
                                      tasks='eval_datasets/FlatWorld-v0/finite_tasks.txt', seed=cur_seed)
        evaluate_deepsets_infinite('FlatWorld-v0', True, 'deepsets_stay',
                                               seed=cur_seed)

        # evaluate_deepsets('FlatWorld-v0', True, 'deepsets_update',
        #                               tasks='eval_datasets/FlatWorld-v0/finite_tasks.txt', seed=cur_seed)
        # evaluate_deepsets_infinite('FlatWorld-v0', True, 'deepsets_update',
        #                            seed=cur_seed)

        print("gnn")

        # evaluate_gnn('gnn_FlatWorld-v0', True, 'gcn_update_2',
        #                          tasks='eval_datasets/FlatWorld-v0/finite_tasks.txt', seed=cur_seed)
        # evaluate_gnn_infinite('gnn_FlatWorld-v0', True, 'gcn_update_2', seed=cur_seed)

        # print("transformer")
        # evaluate_chessworld8_deepsets('big_transformer_ChessWorld-v1', True, 'transformer_stay',
        #                               tasks='eval_datasets/ChessWorld-v1/finite_tasks.txt', seed=cur_seed)
        # evaluate_chessworld8_deepsets_infinite('big_transformer_ChessWorld-v1', True, 'transformer_stay',
        #                                        seed=cur_seed)

    # evaluate_chessworld8_deepsets('big_sets_ChessWorld-v1', True, 'deepsets_trial_3',
    #                               tasks='eval_datasets/ChessWorld-v1/finite_tasks.txt', seed=2)