import os

from train.experiment_metadata import ExperimentMetadata


def get_experiment_path(experiment: ExperimentMetadata) -> str:
    path = f'experiments/{experiment.algorithm}/{experiment.env}/{experiment.name}/{experiment.seed}'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path
