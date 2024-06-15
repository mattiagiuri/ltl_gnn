#!/usr/bin/env fish

conda activate deepltl
PYTHONPATH=src/ ./run_letter.py --num_procs 16 --device cpu --name entropy --seed 1 --log_wandb
PYTHONPATH=src/ ./run_letter.py --num_procs 16 --device cpu --name entropy --seed 2 --log_wandb
PYTHONPATH=src/ ./run_letter.py --num_procs 16 --device cpu --name entropy --seed 3 --log_wandb