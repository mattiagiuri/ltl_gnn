import argparse
import os


def get_experiment_path(env: str, name: str, seed: int) -> str:
    path = f'experiments/ppo/{env}/{name}/{seed}'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def get_experiment_path_from_config(config: argparse.Namespace) -> str:
    experiment = config.experiment
    return get_experiment_path(experiment.env, experiment.name, experiment.seed)


def get_pretraining_experiment_path(env: str, pretraining_experiment: str, seed: int) -> str:
    return f'experiments/ppo/pretraining_{env}/{pretraining_experiment}/{seed}'
