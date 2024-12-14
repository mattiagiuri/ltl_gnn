#!/usr/bin/env python
import json
import os
import subprocess
import sys
from dataclasses import dataclass
import simple_parsing
import wandb

from src.utils.utils import kill_all_wandb_processes


@dataclass
class Args:
    name: str
    seed: int | list[int]
    device: str
    num_procs: int
    log_csv: bool = True
    log_wandb: bool = False
    save: bool = True

# TODO: ask about what happens in each of the 3, Flatword seems normal, LetterEnv has a warning, Zones I modified actor


def main():
    args = simple_parsing.parse(Args)
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src/'
    seeds = args.seed if isinstance(args.seed, list) else [args.seed]
    underlyings = ["FlatWorld-v0", "PointLtl2-v0", "LetterEnv-v0"]
    underlying = underlyings[0]
    for seed in seeds:
        command = [
            'python', 'src/train/train_ppo.py',
            '--env', 'pretraining_'+underlying,
            '--steps_per_process', '512',
            '--batch_size', '1024',
            '--lr', '0.001',
            '--entropy_coef', '0.0',
            '--discount', '0.5',
            '--clip_eps', '0.1',
            '--gae_lambda', '0.5',
            '--log_interval', '1',
            '--save_interval', '1',
            '--epochs', '2',
            '--num_steps', '3_000_000',
            '--model_config', 'pretraining_'+underlying,
            '--curriculum', 'pretraining_'+underlying,
            '--name', args.name,
            '--seed', str(seed),
            '--device', args.device,
            '--num_procs', str(args.num_procs),
        ]
        if args.log_wandb:
            command.append('--log_wandb')
        if not args.log_csv:
            command.append('--no-log_csv')
        if not args.save:
            command.append('--no-save')

        subprocess.run(command, env=env)


if __name__ == '__main__':
    if len(sys.argv) == 1:  # if no arguments are provided, use the following defaults
        sys.argv += '--num_procs 8 --device cpu --name seq --seed 1 --log_csv false --save'.split(' ')
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted!')
        wandb.finish()
        kill_all_wandb_processes()
        sys.exit(0)
