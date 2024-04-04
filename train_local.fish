#!/usr/bin/env fish

conda activate deepltl
./train.sh --num_procs 8 --device cpu --name local --seed 0
