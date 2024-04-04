#!/usr/bin/env bash

usage_msg="Usage: $0 --name <experiment_name> --seed <seed1> <seed2> ... --device <device> --num_procs <num_procs> [--log_wandb]"
experiment_name=""
device=""
num_procs=""
log_wandb=""
declare -a seeds

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --name)
      experiment_name=$2
      shift 2
      ;;
    --seed)
      shift
      while [[ "$#" -gt 0 && "$1" != "--"* ]]; do
        seeds+=("$1")
        shift
      done
      ;;
    --device)
      device=$2
      shift 2
      ;;
    --num_procs)
      num_procs=$2
      shift 2
      ;;
    --log_wandb)
      log_wandb="--log_wandb"
      shift
      ;;
    *)
      echo "Error: Invalid argument"
      echo $usage_msg
      exit 1
  esac
done

# Check if all arguments are provided
if [ -z "$experiment_name" ] || [ ${#seeds[@]} -eq 0 ] || [ -z "$device" ] || [ -z "$num_procs" ]; then
    echo $usage_msg
    exit 1
fi

for seed in "${seeds[@]}"; do
    PYTHONPATH=src/ \
        python src/train/train_ppo.py \
        --env PointLtl2-v0 \
        --ltl_sampler eventually_sampler \
        --steps_per_process 4096 \
        --batch_size 2048 \
        --lr 0.0003 \
        --discount 0.998 \
        --entropy_coef  0.003 \
        --log_interval 1 \
        --save_interval 2 \
        --epochs 10 \
        --num_steps 5000000 \
        --model_config default \
        --name $experiment_name \
        --seed $seed \
        --device $device \
        --num_procs $num_procs \
        $log_wandb
done

exit 0