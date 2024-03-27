import os

import torch

import utils
from train.experiment_metadata import ExperimentMetadata


class ModelStore:
    def __init__(self, experiment: ExperimentMetadata):
        self.path = utils.get_experiment_path(experiment)

    def save_training_status(self, status: dict[str, any]):
        torch.save(status, f'{self.path}/status.pth')

    def save_best_model(self, status: dict[str, any]):
        torch.save(status, f'{self.path}/best_model.pth')

    def load_training_status(self) -> dict[str, any]:
        if not os.path.exists(f'{self.path}/status.pth'):
            raise FileNotFoundError(f'No training status found at {self.path}/status.pth')
        return torch.load(f'{self.path}/status.pth')

    def load_best_model(self) -> dict[str, any]:
        if not os.path.exists(f'{self.path}/best_model.pth'):
            raise FileNotFoundError(f'No best model found at {self.path}/best_model.pth')
        return torch.load(f'{self.path}/best_model.pth')