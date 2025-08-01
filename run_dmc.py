#!/usr/bin/env python
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


def main():
    args = simple_parsing.parse(Args)
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src/'
    seeds = args.seed if isinstance(args.seed, list) else [args.seed]
    for seed in seeds:
        command = [
            'python', 'src/train/train_ppo.py',
            '--env', 'ltl_point_mass',
            '--ltl_sampler', 'eventually_sampler',
            '--steps_per_process', '2048',
            '--batch_size', '1024',
            '--lr', '0.0003',
            '--discount', '0.998',
            '--entropy_coef', '0.003',
            '--log_interval', '1',
            '--save_interval', '2',
            '--epochs', '10',
            '--num_steps', '3_500_000',
            '--model_config', 'default',
            '--name', args.name,
            '--seed', str(seed),
            '--device', args.device,
            '--num_procs', str(args.num_procs),
            # '--pretraining_experiment', 'pretraining',
            # '--freeze_pretrained'
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
        sys.argv += '--num_procs 8 --device cpu --name try --seed 3 --log_csv false --save false'.split(' ')
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted!')
        wandb.finish()
        kill_all_wandb_processes()
        sys.exit(0)
