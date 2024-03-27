from typing import Type

import safety_gymnasium
from gymnasium.wrappers import FlattenObservation

from envs.zones.safety_gym_wrapper import SafetyGymWrapper
from envs.ltl_goal_wrapper import LTLGoalWrapper
from envs.goal_index_wrapper import GoalIndexWrapper
from envs.remove_trunc_wrapper import RemoveTruncWrapper

from ltl import LTLSampler


def make_env(name: str, ltl_sampler: Type[LTLSampler], render_mode: str | None = None):
    env = safety_gymnasium.make(name, render_mode=render_mode)
    env = SafetyGymWrapper(env)
    env = FlattenObservation(env)
    env = LTLGoalWrapper(env, ltl_sampler(env.get_propositions()))
    env = GoalIndexWrapper(env)
    env = RemoveTruncWrapper(env)
    return env
