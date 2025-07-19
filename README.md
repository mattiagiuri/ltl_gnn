# Zero-Shot Instruction Following in RL via Structured LTL Representations

This repository contains the code for the paper "Zero-Shot Instruction Following in RL via Structured LTL Representations" submitted to the ICML workshop "Programmatic Representations for Agent Learning" in 2025. Paper accepted and presented in a poster session at aforementioned workshop. Link: https://openreview.net/forum?id=XDMRtSRnaO.

## Installation
The code requires Python 3.10 with a working installation of PyTorch (tested with version 2.2.2). Use the following command to install the internal packages (from the main directory):
```bash
# activate the virtual environment, e.g. using conda
conda activate your_env
cd src/envs/zones/safety-gymnasium
pip install -e .
cd -
cd src
pip install -e .
cd ..
```
To install the remaining dependencies, run
```bash
pip install -r requirements.txt
```
We use _Rabinizer 4_ (https://www7.in.tum.de/~kretinsk/rabinizer4.html) for the conversion of LTL formulae into LDBAs. This requires Java 11 to be installed on your system and `$JAVA_HOME` to be set accordingly. To test the installation, run
```bash
./rabinizer4/bin/ltl2ldba -h
```
which should print a help message.


## Training

To train a model on an environment, run the `train_ppo.py` file in `src/train`. We provide convenience scripts to train a model with the default parameters in our evaluation environment (_ChessWorld_). To train LTL-GNN, DeepLTL, Transformer, use the commands (respectively)
```bash
python run_chessworld_gnn_stay.py --num_procs 16 --device cpu --name tmp_gnn --seed 1 --log_csv false --save true
python run_chessworld.py --num_procs 16 --device cpu --name tmp_deepsets --seed 1 --log_csv false --save true
python run_transformer.py --num_procs 16 --device cpu --name tmp_transformer --seed 1 --log_csv false --save true
```
The resulting logs and model files will be saved in `experiments/ppo/ChessWorld-v1/tmp_gnn/1` (and similarly for the others), where `ChessWorld-v1` is the internal name for the _ChessWorld_ environment.

## Evaluation

For our evaluation results, we use (respectively for tabular data and our experiment which progressively increases the number of pieces)


```bash
python src/evaluation/evaluate_chessworld8.py
python src/evaluation/chessworld8_stay_ablation.py
```

To produce the graph for the latter evaluation suite, run `plot_ablation_curves.py`. The tables in our paper are available in a relevant jupyter notebook

```bash
jupyter notebook visualize_paper_results_chessworld.ipynb
```

Finally, we provide scripts which evaluate the performance of a model over training on a fixed dataset of specifications from the _reach/avoid_ task space. For LTL-GNN, DeepLTL, Transformer run respectively

```bash
python src/evaluation/chessworld8_eval_over_time.py --model_config big_stay_ChessWorld-v1 --exp tmp_gnn --seed 1
python src/evaluation/chessworld8_eval_over_time_deepsets.py --model_config big_sets_ChessWorld-v1 --exp tmp_deepsets --seed 1
python src/evaluation/chessworld8_eval_over_time_transformer.py --model_config big_transformer_ChessWorld-v1 --exp tmp_transformer --seed 1
```

where "--exp" should be the same as one of the "--name" arguments from the training scripts. To plot the resulting training curves, run `plot_training_curves.py`.
