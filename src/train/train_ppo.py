import argparse

import gymnasium
import simple_parsing
import time
import datetime

import config
import preprocessing
import torch_ac

import utils
from ltl import EventuallySampler
from model.ltl.ltl_embedding import LtlEmbedding
from model.model import Model
from model.policy.continuous_actor import ContinuousActor
from utils import torch_utils
from envs import make_env
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
        self.model_store = ModelStore(args)

    def train(self, log_csv: bool = True, log_wandb: bool = False):
        envs = self.make_envs()
        training_status, resuming = self.get_training_status()
        model = self.build_model(envs[0], training_status)
        algo = torch_ac.PPO(envs, model, self.args.experiment.device, self.args.ppo,
                            preprocess_obss=preprocessing.preprocess_obss)
        if "optimizer_state" in training_status:
            algo.optimizer.load_state_dict(training_status["optimizer_state"])
            self.text_logger.info("Loaded optimizer from existing run.\n")
        logger = self.make_logger(log_csv, log_wandb, resuming)
        logger.log_config()

        num_steps = training_status["num_steps"]
        num_updates = training_status["num_updates"]
        while num_steps < self.args.experiment.num_steps:
            start = time.time()
            exps, logs = algo.collect_experiences()
            update_logs = algo.update_parameters(exps)
            logs.update(update_logs)
            update_time = time.time() - start

            num_steps += logs["num_steps"]
            num_updates += 1
            if num_updates % self.args.experiment.log_interval == 0:
                logs = self.augment_logs(logs, update_time, num_steps)
                logger.log(logs)
            if self.args.experiment.save_interval > 0 and num_updates % self.args.experiment.save_interval == 0:
                training_status = {"num_steps": num_steps, "num_updates": num_updates,
                                   "model_state": algo.model.state_dict(),
                                   "optimizer_state": algo.optimizer.state_dict()}
                self.model_store.save_training_status(training_status)
                self.text_logger.info("Saved training status")

    def make_envs(self) -> list[gymnasium.Env]:
        utils.set_seed(self.args.experiment.seed)
        envs = []
        for i in range(self.args.experiment.num_procs):
            assert self.args.experiment.ltl_sampler == 'eventually_sampler'
            envs.append(make_env(self.args.experiment.env, EventuallySampler))
        # TODO: sample random seeds for the different environments
        envs[0].reset(seed=self.args.experiment.seed)
        self.text_logger.info("Environments loaded.\n")
        return envs

    def get_training_status(self) -> tuple[dict, bool]:
        resuming = False
        try:
            training_status = self.model_store.load_training_status()
            self.text_logger.info("Resuming training from existing run.\n")
            resuming = True
        except FileNotFoundError:
            training_status = {"num_steps": 0, "num_updates": 0}
        return training_status, resuming

    # noinspection PyShadowingNames
    def build_model(self, env: gymnasium.Env, training_status: dict) -> Model:
        model_config = model_configs[self.args.model_config]
        obs_dim = env.observation_space['features'].shape[0]
        action_dim = env.action_space.shape[0]
        env_net = torch_utils.make_mlp_layers([obs_dim, *model_config.env_net.layers],
                                              activation=model_config.env_net.activation)
        env_embedding_dim = model_config.env_net.layers[-1]
        ltl_embedding_dim = 32
        ltl_net = LtlEmbedding(5, ltl_embedding_dim)
        actor = ContinuousActor(action_dim=action_dim,
                                layers=[env_embedding_dim + ltl_embedding_dim, *model_config.actor.layers],
                                activation=model_config.actor.activation,
                                state_dependent_std=model_config.actor.state_dependent_std)
        critic = torch_utils.make_mlp_layers([env_embedding_dim + ltl_embedding_dim, *model_config.critic.layers, 1],
                                             activation=model_config.critic.activation,
                                             final_layer_activation=False)
        model = Model(env_net, ltl_net, actor, critic)
        if "model_state" in training_status:
            model.load_state_dict(training_status["model_state"])
            self.text_logger.info("Loaded model from existing run.\n")
        model.to(self.args.experiment.device)
        self.text_logger.info(f'Num parameters: {torch_utils.get_number_of_params(model)}\n')
        return model

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
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    trainer = Trainer(args)
    trainer.train(log_csv=True, log_wandb=False)


if __name__ == '__main__':
    main()
