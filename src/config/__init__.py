from .experiment_config import *
from .ppo_config import *
from .model_config import *

model_configs = {
    'default': default_model_config,
    'point_mass': point_mass,
}

__all__ = ['ExperimentConfig', 'PPOConfig', 'ModelConfig', 'model_configs']
