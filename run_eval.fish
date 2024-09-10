#!/bin/env fish

conda activate deepltl

set seeds 1 2 3 4 5

for seed in $seeds
    PYTHONPATH=src/ python src/evaluation/eval.py --seed $seed
end