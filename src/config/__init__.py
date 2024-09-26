from .experiment_config import *
from .ppo_config import *
from .model_config import *

model_configs = {
    'PointLtl2-v0': zones,
    'LetterEnv-v0': letter,
    'FlatWorld-v0': flatworld,
}

__all__ = ['ExperimentConfig', 'PPOConfig', 'ModelConfig', 'SetNetConfig', 'model_configs']
