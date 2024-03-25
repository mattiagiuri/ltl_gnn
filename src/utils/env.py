import safety_gymnasium

from envs.ltl2action_wrapper import Ltl2ActionWrapper
from envs.ltl_wrapper import LtlWrapper
from envs.zones.safety_gym_wrapper.safety_gym_wrapper import SafetyGymWrapper
from ltl_wrappers import LTLEnv

def make_env(name: str, ltl_sampler, seed=None, intrinsic=0):
    env = safety_gymnasium.make(name)
    env = SafetyGymWrapper(env)
    env = LtlWrapper(env)
    env = Ltl2ActionWrapper(env)
    env = LTLEnv(env, 'full', ltl_sampler, intrinsic)
    return env
