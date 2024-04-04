from typing import Type

from gymnasium.wrappers import FlattenObservation

from envs.zones.safety_gym_wrapper import SafetyGymWrapper
from envs.ltl_goal_wrapper import LTLGoalWrapper
from envs.goal_index_wrapper import GoalIndexWrapper
from envs.remove_trunc_wrapper import RemoveTruncWrapper

from ltl import LTLSampler


def make_env(name: str, ltl_sampler: Type[LTLSampler], render_mode: str | None = None):
    if is_safety_gym_env(name):
        return make_safety_gym_env(name, ltl_sampler, render_mode)
    else:
        return make_dmc_env(name, ltl_sampler, render_mode)


def is_safety_gym_env(name: str) -> bool:
    return any([name.startswith(agent_name) for agent_name in ['Point', 'Car', 'Racecar', 'Doggo', 'Ant']])


def make_safety_gym_env(name: str, ltl_sampler: Type[LTLSampler], render_mode: str | None = None):
    import safety_gymnasium
    env = safety_gymnasium.make(name, render_mode=render_mode)
    env = SafetyGymWrapper(env)
    env = FlattenObservation(env)
    env = LTLGoalWrapper(env, ltl_sampler(env.get_propositions()))
    env = GoalIndexWrapper(env)
    env = RemoveTruncWrapper(env)
    return env


def make_dmc_env(name: str, ltl_sampler: Type[LTLSampler], render_mode: str | None = None):
    return {}