# DeepLTL: Learning to Efficiently Satisfy Complex LTL Specifications

This repository contains the code for the paper "DeepLTL: Learning to Efficiently Satisfy Complex LTL Specifications" submitted to ICLR 2025.

## Installation
The code requires Python 3.10 with a working installation of PyTorch (tested with version 2.2.2). In order to use the _ZoneEnv_ environment, use the following command to install the required dependencies:
```bash
# activate the virtual environment, e.g. using conda
conda activate deepltl
cd src/envs/zones/safety-gymnasium
pip install -e .
```
This will also take care of installing the required versions of `mujoco`, `gymnasium` and `gymnasium-robotics`. To install the remaining dependencies, run
```bash
pip install -r requirements.txt
```
We use _Rabinizer 4_ (https://www7.in.tum.de/~kretinsk/rabinizer4.html) for the conversion of LTL formulae into LDBAs. This requires Java 11 to be installed on your system and `$JAVA_HOME` to be set accordingly. To test the installation, run
```bash
./rabinizer4/bin/ltl2ldba -h
```
which should print a help message.

## Training

To train a model on an environment, run the `train_ppo.py` file in `src/train`. We provide convenience scripts to train a model with the default parameters in our evaluation environments (_LetterWorld_, _ZoneEnv_, and _FlatWorld_). For example, to train a model on the _ZoneEnv_ environment, run
```bash
PYTHONPATH=src/ python run_zones.py --device cpu --name test --seed 1
```
The resulting logs and model files will be saved in `experiments/ppo/PointLtl2-v0/test/1` (where `PointLtl2-v0` is the internal name for the _ZoneEnv_ environment).

## Evaluation

We provide several evaluation scripts in `src/evaluation`. To simulate a trained model with a given LTL formula and output several statistics such as success rate (SR) and average number of steps (μ), run
```bash
PYTHONPATH=src/ python src/evaluation/simulate.py --env PointLtl2-v0 --exp test --seed 1 --formula "(!blue U green) & F yellow" --finite --deterministic
```
The script also supports a `--render` flag to visualise the simulation in real time. Alternatively, we provide the scripts `draw_zone_trajectories.py` and `draw_flat_trajectories.py` to visualise the trajectories of the agents in the _ZoneEnv_ and _FlatWorld_ environments, respectively.

For a more comprehensive evaluation, we provide the scripts `eval_test_tasks_finite.py` and `eval_test_tasks_infinite.py` to evaluate the performance of a model on a set of test tasks. The former evaluates the model on a set of finite-horizon tasks, while the latter evaluates the model on a set of infinite-horizon tasks. The default tasks specified in the scripts match the tasks from our evaluation in Table 1.

Finally, we provide the script `eval_over_time.py` which evaluates the performance of a model over training on a fixed dataset of specifications sampled from the _reach/avoid_ task space. To plot the resulting training curves, run `plot_training_curves.py`.
