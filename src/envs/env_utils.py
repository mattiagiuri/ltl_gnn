from typing import Type, Callable

from gymnasium.wrappers import FlattenObservation, TimeLimit

from envs.alternate_wrapper import AlternateWrapper
from envs.dict_wrapper import DictWrapper
from envs.ldba_graph_wrapper import LDBAGraphWrapper
from envs.ltl_goal_wrapper import LTLGoalWrapper
from envs.remove_trunc_wrapper import RemoveTruncWrapper

from ltl import LTLSampler
from ltl.logic import FrozenAssignment


def make_env(name: str, make_sampler: Callable[[list[str]], LTLSampler], render_mode: str | None = None):
    if name.startswith('pretraining_'):
        underlying = name[len('pretraining_'):]
        underlying_env = make_env(underlying, make_sampler, render_mode)
        propositions = underlying_env.get_wrapper_attr('get_propositions')()
        impossible_assignments = underlying_env.get_wrapper_attr('get_impossible_assignments')()
        return make_pretraining_env(propositions, impossible_assignments, make_sampler)
    if is_safety_gym_env(name):
        return make_safety_gym_env(name, make_sampler, render_mode)
    else:
        return make_dmc_env(name, make_sampler, render_mode)


def is_safety_gym_env(name: str) -> bool:
    return any([name.startswith(agent_name) for agent_name in ['Point', 'Car', 'Racecar', 'Doggo', 'Ant']])


def make_safety_gym_env(name: str, ltl_sampler: Callable[[list[str]], LTLSampler], render_mode: str | None = None):
    # noinspection PyUnresolvedReferences
    import safety_gymnasium
    from envs.zones.safety_gym_wrapper import SafetyGymWrapper

    env = safety_gymnasium.make(name, render_mode=render_mode)
    env = SafetyGymWrapper(env)
    env = FlattenObservation(env)
    env = LTLGoalWrapper(env, ltl_sampler(env.get_wrapper_attr('get_propositions')()))
    # env = GoalIndexWrapper(env, punish_termination=True)
    env = TransitionGraphWrapper(env, punish_termination=True)
    env = RemoveTruncWrapper(env)
    return env


def make_dmc_env(name: str, ltl_sampler: Callable[[list[str]], LTLSampler], render_mode: str | None = None):
    # noinspection PyUnresolvedReferences
    from dm_control import suite, viewer
    import envs.dmc as dmc
    dmc.register_with_suite()
    from envs.dmc.dmc_gym_wrapper.dmc_gym_wrapper import DMCGymWrapper

    env = suite.load(domain_name=name, task_name='ltl', visualize_reward=False)
    env = DMCGymWrapper(env, render_mode=render_mode)
    env = FlattenObservation(env)
    if name.startswith('ltl_cartpole'):
        # load alternate task
        env = DictWrapper(env)
        env = AlternateWrapper(env, ['yellow', 'green'])
        env = RemoveTruncWrapper(env)
    else:
        env = LTLGoalWrapper(env, ltl_sampler(env.get_wrapper_attr('get_propositions')()))
        # env = GoalIndexWrapper(env, punish_termination=False)
        env = LDBAGraphWrapper(env, punish_termination=True)
        env = RemoveTruncWrapper(env)
    return env


def make_pretraining_env(
        propositions: set[str],
        impossible_assignments: set[FrozenAssignment],
        ltl_sampler: Callable[[list[str]], LTLSampler]
):
    from envs.pretraining.pretraining_env import PretrainingEnv

    env = PretrainingEnv(propositions, impossible_assignments)
    env = LTLGoalWrapper(env, ltl_sampler(env.get_wrapper_attr('get_propositions')()))
    env = LDBAGraphWrapper(env, punish_termination=True)
    env = TimeLimit(env, max_episode_steps=100)
    env = RemoveTruncWrapper(env)
    return env
