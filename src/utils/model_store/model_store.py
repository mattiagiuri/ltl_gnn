import argparse
import os

import torch

import utils


class ModelStore:
    def __init__(self, config: argparse.Namespace):
        self.path = utils.get_experiment_path(config)
        self.pretraining_experiment_path = utils.get_pretraining_experiment_path(config)

    def save_training_status(self, status: dict[str, any]):
        torch.save(status, f'{self.path}/status.pth')

    def save_ltl_net(self, ltl_net: dict[str, any]):
        torch.save(ltl_net, f'{self.path}/ltl_net.pth')

    def save_best_model(self, status: dict[str, any]):
        torch.save(status, f'{self.path}/best_model.pth')

    def load_training_status(self) -> dict[str, any]:
        if not os.path.exists(f'{self.path}/status.pth'):
            raise FileNotFoundError(f'No training status found at {self.path}/status.pth')
        return torch.load(f'{self.path}/status.pth')

    def load_pretrained(self) -> dict[str, any]:
        if not os.path.exists(f'{self.pretraining_experiment_path}/ltl_net.pth'):
            raise FileNotFoundError(f'No pretrained model found at {self.pretraining_experiment_path}/ltl_net.pth')
        return torch.load(f'{self.pretraining_experiment_path}/ltl_net.pth')

    def load_best_model(self) -> dict[str, any]:
        if not os.path.exists(f'{self.path}/best_model.pth'):
            raise FileNotFoundError(f'No best model found at {self.path}/best_model.pth')
        return torch.load(f'{self.path}/best_model.pth')
