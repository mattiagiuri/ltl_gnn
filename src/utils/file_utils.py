import argparse
import os


def get_experiment_path(config: argparse.Namespace) -> str:
    experiment = config.experiment
    path = f'experiments/ppo/{experiment.env}/{experiment.name}/{experiment.seed}'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path
