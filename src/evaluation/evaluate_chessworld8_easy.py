from evaluation.simulate import simulate
from envs.chessworld import ChessWorld
import pandas as pd
import re
from preprocessing.vocab import init_vocab, init_vars
import os

tasks_path = "eval_datasets/ChessWorld-v2/tasks.txt"
env = 'ChessWorld-v2'
exp_gnn = "gcn"
exp_gnn_prop = "gcn_prop"
exp_gnn_fine = "gcn_fine"
exp_deepsets = "deepsets_full"
exp_deepsets_prop = "deepsets_prop"
seed = 1
num_episodes = 30
gamma = 0.98
finite = True
render = False
deterministic = True
gnn_gnn = True
gnn_deepsets=False
init_voc = True

os.makedirs("results_chessworld/ChessWorld-v2", exist_ok=True)



def read_tasks():
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


def evaluate_chessworld8_gnn(save=False, exp_gnn=exp_gnn):
    global init_voc
    task_list = read_tasks()
    results = {"Task Set": [], "Task ID": [], "Successes x/30": [], "Avg Steps": [], "Avg Discounted Return": []}

    for task_set, tasks in task_list.items():
        for name, task in tasks:
            successes, avg_steps, adr = simulate(env, gamma, exp_gnn, seed, num_episodes, task, finite,
                                                 render, deterministic, gnn_gnn, init_voc)
            if init_voc:
                init_voc = False

            results["Task Set"].append(task_set)
            results["Task ID"].append(name)
            results["Successes x/30"].append(successes)
            results["Avg Steps"].append(avg_steps)
            results["Avg Discounted Return"].append(adr)

    df_results = pd.DataFrame(results)
    df_results.set_index(["Task Set", "Task ID"], inplace=True)

    if save:
        df_results.to_csv("results_chessworld/ChessWorld-v2/" + env + "_" + exp_gnn + ".csv")

    return df_results


def evaluate_chessworld8_deepsets(save=False, exp_deepsets=exp_deepsets):
    task_list = read_tasks()
    results = {"Task Set": [], "Task ID": [], "Successes x/30": [], "Avg Steps": [], "Avg Discounted Return": []}

    for task_set, tasks in task_list.items():
        for name, task in tasks:
            successes, avg_steps, adr = simulate(env, gamma, exp_deepsets, seed, num_episodes, task, finite,
                                                 render, deterministic, gnn_deepsets)

            results["Task Set"].append(task_set)
            results["Task ID"].append(name)
            results["Successes x/30"].append(successes)
            results["Avg Steps"].append(avg_steps)
            results["Avg Discounted Return"].append(adr)

    df_results = pd.DataFrame(results)
    df_results.set_index(["Task Set", "Task ID"], inplace=True)

    if save:
        df_results.to_csv("results_chessworld/ChessWorld-v2/" + env + "_" + exp_deepsets + ".csv")

    return df_results


if __name__ == "__main__":

    evaluate_chessworld8_deepsets(True)
    evaluate_chessworld8_deepsets(True, exp_deepsets_prop)
    evaluate_chessworld8_gnn(True, exp_gnn)
    # evaluate_chessworld_gnn(True, exp_gnn_fine)
    # evaluate_chessworld_gnn(True, exp_gnn_prop)
    # evaluate_chessworld_gnn(True, "gcn_prop_fine")
