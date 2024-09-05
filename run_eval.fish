#!/bin/env fish

conda activate deepltl

set seeds 1 2

for seed in $seeds
    PYTHONPATH=src/ python src/eval/eval.py --seed $seed
end