#!/usr/bin/env fish

conda activate deepltl
set device gpu
set name eval
set num_procs 16
set seeds 1 2

for seed in $seeds
    PYTHONPATH=src/ ./run_zones.py --num_procs $num_procs --device $device --name $name --seed $seed --log_wandb
end
