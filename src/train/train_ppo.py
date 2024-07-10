import argparse
from typing import Optional

import gymnasium
import simple_parsing
import time
import datetime

import torch

import config
import preprocessing
import torch_ac

import utils
from model.model import build_model
from envs import make_env, get_env_attr
from sequence import CurriculumSequenceSampler
from sequence.random_curriculum_sampler import RandomCurriculumSampler
from utils import torch_utils
from utils.logging.file_logger import FileLogger
from utils.logging.multi_logger import MultiLogger
from utils.logging.text_logger import TextLogger
from utils.logging.wandb_logger import WandbLogger
from utils.model_store import ModelStore
from config import *


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.text_logger = TextLogger(args)
        self.model_store = ModelStore.from_config(args)

    def train(self, log_csv: bool = True, log_wandb: bool = False):
        envs = self.make_envs()
        training_status, resuming = self.get_training_status()
        pretrained_model = self.load_pretrained_model()
        model = build_model(envs[0], training_status, model_configs[self.args.model_config])
        model.to(self.args.experiment.device)
        print(model.ltl_net)
        algo = torch_ac.PPO(envs, model, self.args.experiment.device, self.args.ppo,
                            preprocess_obss=preprocessing.preprocess_obss,
                            parallel=False)  # TODO: add cmd argument for this
        if "optimizer_state" in training_status:
            algo.optimizer.load_state_dict(training_status["optimizer_state"])
            self.text_logger.info("Loaded optimizer from existing run.")
        logger = self.make_logger(log_csv, log_wandb, resuming)
        logger.log_config()

        self.text_logger.info(f'Num parameters: {torch_utils.get_number_of_params(model)}')
        num_steps = training_status["num_steps"]
        num_updates = training_status["num_updates"]
        while num_steps < self.args.experiment.num_steps:
            start = time.time()
            exps, logs = algo.collect_experiences()
            for env in envs:
                seq_sampler = get_env_attr(env, 'sample_sequence')
                if seq_sampler.is_adaptive:
                    seq_sampler.update_goal_success(logs['avg_goal_success'])
            update_logs = algo.update_parameters(exps)
            logs.update(update_logs)
            update_time = time.time() - start

            num_steps += logs["num_steps"]
            num_updates += 1
            if num_updates % self.args.experiment.log_interval == 0:
                logs = self.augment_logs(logs, update_time, num_steps)
                logger.log(logs)
            if self.args.experiment.save_interval > 0 and num_updates % self.args.experiment.save_interval == 0 \
                    and self.args.save:
                training_status = {"num_steps": num_steps, "num_updates": num_updates,
                                   "model_state": algo.model.state_dict(),
                                   "optimizer_state": algo.optimizer.state_dict()}
                self.model_store.save_training_status(training_status)
                self.model_store.save_ltl_net(algo.model.ltl_net.state_dict())
                self.text_logger.info("Saved training status")

    def make_envs(self) -> list[gymnasium.Env]:
        utils.set_seed(self.args.experiment.seed)
        envs = []
        for i in range(self.args.experiment.num_procs):
            # sampler = CurriculumSequenceSampler.partial()
            sampler = RandomCurriculumSampler.partial()
            envs.append(make_env(self.args.experiment.env, sampler, sequence=True))
        # Set different seeds for each environment. The seed offset is used to ensure that the seeds do not overlap.
        seed_offset = 100 * self.args.experiment.seed
        seeds = [seed_offset + i for i in range(self.args.experiment.num_procs)]
        self.text_logger.info(f"Using seeds: {seeds}")
        for env, seed in zip(envs, seeds):
            env.reset(seed=seed)
        self.text_logger.info("Environments loaded.")
        return envs

    def get_training_status(self) -> tuple[dict, bool]:
        resuming = False
        try:
            training_status = self.model_store.load_training_status()
            self.text_logger.important_info("Resuming training from existing run.")
            resuming = True
        except FileNotFoundError:
            training_status = {"num_steps": 0, "num_updates": 0}
        return training_status, resuming

    def load_pretrained_model(self) -> Optional[dict]:
        if self.args.pretraining_experiment is not None:
            pretrained = self.model_store.load_pretrained()
            self.text_logger.important_info(f"Loaded pretrained model.")
            return pretrained
        return None

    def make_logger(self, log_csv: bool, log_wandb: bool, resuming: bool) -> MultiLogger:
        loggers = [self.text_logger]
        if log_csv:
            loggers.append(FileLogger(self.args, resuming=resuming))
        if log_wandb:
            loggers.append(WandbLogger(self.args, project_name='deep-ltl', resuming=resuming))
        return MultiLogger(*loggers)

    def augment_logs(self, logs: dict, update_time: float, num_steps: int) -> dict:
        sps = logs["num_steps"] / update_time
        remaining_duration = int((self.args.experiment.num_steps - num_steps) / sps)
        remaining_duration = 0 if remaining_duration < 0 else remaining_duration
        remaining_time = str(datetime.timedelta(seconds=remaining_duration))

        average_reward_per_step = utils.average_reward_per_step(logs["return_per_episode"],
                                                                logs["num_steps_per_episode"])
        average_discounted_return = utils.average_discounted_return(logs["return_per_episode"],
                                                                    logs["num_steps_per_episode"],
                                                                    self.args.ppo.discount)
        logs.update({
            "arps": average_reward_per_step,
            "adr": average_discounted_return,
            'sps': sps,
            'remaining': remaining_time,
            'num_steps': num_steps  # set num_steps to the total number of steps
        })
        return logs


# noinspection PyTypeChecker
def parse_arguments() -> argparse.Namespace:
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(config.ExperimentConfig, dest="experiment")
    parser.add_arguments(config.PPOConfig, dest="ppo")
    parser.add_argument("--model_config", type=str, default="default", choices=model_configs.keys(),
                        required=True)
    parser.add_argument("--pretraining_experiment", type=str, default=None)
    parser.add_argument("--freeze_pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log_csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.experiment.device == 'gpu':
        assert torch.cuda.is_available(), "CUDA is not available."
        args.experiment.device = 'cuda'

    if args.pretraining_experiment is None and args.freeze_pretrained:
        raise ValueError("Cannot freeze without providing a pretrained model.")
    return args


def main():
    args = parse_arguments()
    trainer = Trainer(args)
    start_time = time.time()
    trainer.train(log_csv=args.log_csv, log_wandb=args.log_wandb)
    training_time = datetime.timedelta(seconds=int(time.time() - start_time))
    print(f"Training took {training_time}.")


if __name__ == '__main__':
    main()
