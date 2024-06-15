from .experiment_config import *
from .ppo_config import *
from .model_config import *

model_configs = {
    'default': default,
    'letter': letter,
    'pretraining': pretraining,
}

__all__ = ['ExperimentConfig', 'PPOConfig', 'ModelConfig', 'model_configs']
