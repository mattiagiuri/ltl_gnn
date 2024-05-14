#!/usr/bin/env python
import json
import os
import subprocess
import sys
from dataclasses import dataclass
import simple_parsing
import wandb

from model.ltl.batched_sequence import BatchedSequence
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
            '--env', 'pretraining_PointLtl2-v0',
            '--ltl_sampler', 'reach4',
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
            '--model_config', 'pretraining',
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


if __name__ == '__main__':  # TODO: make sure that pretraining converges.
    if len(sys.argv) == 1:  # if no arguments are provided, use the following defaults
        sys.argv += '--num_procs 8 --device cpu --name rnn4 --seed 1 --log_csv false --save true'.split(' ')
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted!')
        wandb.finish()
        kill_all_wandb_processes()
        # with open('vocab.json', 'w+') as f:
        #     json.dump(BatchedSequence.VOCAB, f, indent=2)
        sys.exit(0)
