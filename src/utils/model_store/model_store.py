import argparse
import json
import os

import torch

import utils


class ModelStore:
    def __init__(self, env: str, name: str, seed: int, pretraining_experiment: str | None):
        self.path = utils.get_experiment_path(env, name, seed)
        if pretraining_experiment:
            self.pretraining_experiment_path = utils.get_pretraining_experiment_path(env, pretraining_experiment, seed)
        else:
            self.pretraining_experiment_path = None

    @classmethod
    def from_config(cls, config: argparse.Namespace) -> 'ModelStore':
        exp = config.experiment
        return cls(exp.env, exp.name, exp.seed, config.pretraining_experiment)

    def save_training_status(self, status: dict[str, any]):
        torch.save(status, f'{self.path}/status.pth')

    def save_eval_training_status(self, status: dict[str, any]):
        torch.save(status, f'{self.path}/eval/{status["num_steps"]}.pth')

    def save_ltl_net(self, ltl_net: dict[str, any]):
        torch.save(ltl_net, f'{self.path}/ltl_net.pth')

    def save_best_model(self, status: dict[str, any]):
        torch.save(status, f'{self.path}/best_model.pth')

    def load_training_status(self, map_location=None) -> dict[str, any]:
        if not os.path.exists(f'{self.path}/status.pth'):
            raise FileNotFoundError(f'No training status found at {self.path}/status.pth')
        return torch.load(f'{self.path}/status.pth', map_location=map_location)

    def load_pretrained(self) -> dict[str, any]:
        if not self.pretraining_experiment_path:
            raise ValueError('No pretraining experiment provided.')
        if not os.path.exists(f'{self.pretraining_experiment_path}/ltl_net.pth'):
            raise FileNotFoundError(f'No pretrained model found at {self.pretraining_experiment_path}/ltl_net.pth')
        return torch.load(f'{self.pretraining_experiment_path}/ltl_net.pth')

    def load_best_model(self) -> dict[str, any]:
        if not os.path.exists(f'{self.path}/best_model.pth'):
            raise FileNotFoundError(f'No best model found at {self.path}/best_model.pth')
        return torch.load(f'{self.path}/best_model.pth')

    def load_eval_training_statuses(self, map_location=None) -> list[dict[str, any]]:
        eval_dir = f'{self.path}/eval'
        if not os.path.exists(eval_dir):
            raise FileNotFoundError(f'No eval models found at {eval_dir}')
        eval_models = []
        for file in os.listdir(eval_dir):
            eval_models.append(torch.load(f'{eval_dir}/{file}', map_location=map_location))
        final_model = self.load_training_status(map_location)
        eval_models.append(final_model)
        return sorted(eval_models, key=lambda x: x['num_steps'])

