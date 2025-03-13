from .experiment_config import *
from .ppo_config import *
from .model_config import *

model_configs = {
    'PointLtl2-v0': zones,
    'LetterEnv-v0': letter,
    'FlatWorld-v0': flatworld,
    # "pretraining_PointLtl2-v0": pretraining_zones,
    # "pretraining_LetterEnv-v0": pretraining_letter,
    'pretraining_FlatWorld-v0': pretraining_flatworld,
    'pretraining_context_FlatWorld-v0': pretraining_context_flatworld,
    'ChessWorld-v0': chessworld,
    'pretraining_ChessWorld-v0': pretraining_chessworld,
    'gnn_ChessWorld-v0': chessworld_gnn,
    'gnn_train_ChessWorld-v0': chessworld_gnn_train,
    'gnn_ChessWorld-v1': chessworld_gnn,
    'ChessWorld-v1': chessworld,
    'pretraining_ChessWorld-v1': pretraining_chessworld,
    'gnn_train_ChessWorld-v1': chessworld_gnn_train,
    'pretraining_stay_ChessWorld-v1': pretraining_chessworld_stay,
    'stay_ChessWorld-v1': chessworld_gnn_stay,
    'final_stay_ChessWorld-v1': chessworld_gnn_stay_fine,
    'big_stay_ChessWorld-v1': chessworld_gnn_big,
    'big_ChessWorld-v1': chessworld_gnn_big_frozen,
    'big_sets_ChessWorld-v1': chessworld_deepsets_big,
    'big_transformer_ChessWorld-v1': chessworld_transformer_big,
    'frozen_transformer_ChessWorld-v1': chessworld_transformer_frozen,
    'big_prop_ChessWorld-v1': chessworld_gnn_big_prop,
    'frozen_big_prop_ChessWorld-v1': chessworld_gnn_big_frozen_prop,
    'big_gnn_FlatWorld-v0': flatworld_gnn_big,
    'frozen_big_gnn_FlatWorld-v0': flatworld_gnn_big_frozen,
    'big_deepsets_FlatWorld-v0': flatworld_deepsets_big,
    'gnn_FlatWorld-v0': flatworld_gnn,
    'frozen_gnn_FlatWorld-v0': frozen_flatworld_gnn
}

__all__ = ['ExperimentConfig', 'PPOConfig', 'ModelConfig', 'SetNetConfig', 'model_configs']
