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
    'Flatworld-gnn': flatworld_gnn
}

__all__ = ['ExperimentConfig', 'PPOConfig', 'ModelConfig', 'SetNetConfig', 'model_configs']
