#!/usr/bin/env fish

conda activate deepltl
set device gpu
set name stage3_095
set num_procs 16
PYTHONPATH=src/ ./run_letter.py --num_procs $num_procs --device $device --name $name --seed 1 --log_wandb
PYTHONPATH=src/ ./run_letter.py --num_procs $num_procs --device $device --name $name --seed 2 --log_wandb
PYTHONPATH=src/ ./run_letter.py --num_procs $num_procs --device $device --name $name --seed 3 --log_wandb