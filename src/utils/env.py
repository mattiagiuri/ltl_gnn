"""
This class defines the environments that we are going to use.
Note that this is the place to include the right LTL-Wrapper for each environment.
"""


import safety_gymnasium

from envs.ltl2action_wrapper import Ltl2ActionWrapper
from envs.ltl_wrapper import LtlWrapper
from envs.zones.safety_gym_wrapper.safety_gym_wrapper import SafetyGymWrapper
from src.ltl_wrappers import LTLEnv

def make_env(env_key, progression_mode, ltl_sampler, seed=None, intrinsic=0, noLTL=False):
    assert not noLTL

    env = safety_gymnasium.make(env_key)
    env = SafetyGymWrapper(env)
    env = LtlWrapper(env)
    env = Ltl2ActionWrapper(env)
    env = LTLEnv(env, progression_mode, ltl_sampler, intrinsic)
    return env
