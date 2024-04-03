#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <experiment_name> <seed>"
    exit 1
fi

experiment_name=$1
seed=$2

PYTHONPATH=src/ python src/train/train_ppo.py --env PointLtl2-v0 --ltl_sampler eventually_sampler --steps_per_process 4096 --batch_size 2048 --lr 0.0003 --discount 0.998 --entropy_coef  0.003 --log_interval 1 --save_interval 2 --epochs 10 --num_steps 5000000 --model_config default --name $experiment_name --seed $seed

exit 0