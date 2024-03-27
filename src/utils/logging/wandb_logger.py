import dataclasses
import os

import wandb

import utils
from train.experiment_metadata import ExperimentMetadata
from utils.logging.logger import Logger


class WandbLogger(Logger):
    """
    A logger that logs to Weights & Biases.
    """

    WANDB_FILE_NAME = 'wandb_id.txt'

    def __init__(self, experiment: ExperimentMetadata, project_name: str, resuming: bool = False):
        super().__init__(experiment)
        self.project_name = project_name
        self.metadata = experiment
        self.run_id = None
        if resuming:
            wandb_id_file = f'{utils.get_experiment_path(experiment)}/{self.WANDB_FILE_NAME}'
            if not os.path.exists(wandb_id_file):
                raise FileNotFoundError(f'Trying to resume, but no wandb_id.txt file found in {wandb_id_file}.')
            with open(wandb_id_file, 'r') as f:
                self.run_id = f.read().strip()
        self.log_metadata()

    def log_metadata(self):
        if self.run_id is not None:
            wandb.init(
                project=self.project_name,
                id=self.run_id,
                resume='must',
            )
        else:
            run = wandb.init(
                project=self.project_name,
                config=dataclasses.asdict(self.metadata),
            )
            wandb_id_file = f'{utils.get_experiment_path(self.metadata)}/{self.WANDB_FILE_NAME}'
            with open(wandb_id_file, 'w') as f:
                f.write(run.id)

    def log(self, data: dict[str, float | list[float]]):
        data = self.aggregate(data)
        self.check_keys_valid(data)
        wandb.log(data)

    def finish(self):
        wandb.finish()
