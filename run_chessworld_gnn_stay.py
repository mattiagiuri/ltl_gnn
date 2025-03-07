#!/usr/bin/env python
import os
import subprocess
import sys
from dataclasses import dataclass
import simple_parsing
import wandb

from utils import kill_all_wandb_processes


@dataclass
class Args:
    name: str
    seed: int | list[int]
    device: str
    num_procs: int = 16
    log_csv: bool = True
    log_wandb: bool = False
    save: bool = True


def main():
    args = simple_parsing.parse(Args)
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src/'
    seeds = args.seed if isinstance(args.seed, list) else [args.seed]
    for seed in seeds:
        prof_file = 'profile_' + str(seed) + '.prof'
        command = [
            'python',
            # '-m', 'cProfile', '-o', prof_file,
            'src/train/train_ppo.py',
            '--env', 'ChessWorld-v1',
            '--steps_per_process', '2048',  # 1024
            '--epochs', '10',
            '--batch_size', '4096',  # 64
            '--discount', '0.98',
            '--gae_lambda', '0.95',
            '--entropy_coef', '0.003',  # 0.003
            '--log_interval', '1',
            '--save_interval', '1',
            '--num_steps', '15_000_000',
            '--model_config', 'big_ChessWorld-v1',
            '--curriculum', 'formula_ChessWorld-v1',
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
        # change --name tmp to --name whatever_i_want
        sys.argv += '--num_procs 16 --device cpu --name gcn_formula_big_skip_6_finer_p --seed 2 --log_csv false --save true'.split(' ')
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted!')
        wandb.finish()
        # kill_all_wandb_processes()
        sys.exit(0)
    # finally:
    #     print("profile file")
