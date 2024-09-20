from .experiment_config import *
from .ppo_config import *
from .model_config import *

model_configs = {
    'PointLtl2-v0': zones,
    'LetterEnv-v0': letter,
    'FlatWorld-v0': flatworld,
    'pretraining': pretraining,
}

__all__ = ['ExperimentConfig', 'PPOConfig', 'ModelConfig', 'model_configs']
