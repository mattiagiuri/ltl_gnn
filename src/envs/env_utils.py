from typing import Type, Callable, Optional

from gymnasium.wrappers import FlattenObservation, TimeLimit

from envs.alternate_wrapper import AlternateWrapper
from envs.dict_wrapper import DictWrapper
from envs.ltl_goal_wrapper import LTLGoalWrapper
from envs.remove_trunc_wrapper import RemoveTruncWrapper

from ltl import LTLSampler
from ltl.logic import FrozenAssignment


def get_env_attr(env, attr: str):
    if hasattr(env, attr):
        return getattr(env, attr)
    if hasattr(env, 'env'):
        return getattr(env.unwrapped, attr)
    else:
        raise AttributeError(f'Attribute {attr} not found in env.')


def make_ltl_env(
        name: str,
        make_sampler: Callable[[list[str]], LTLSampler],
        max_steps: Optional[int] = None,  # TODO: add max_steps parameter to other environments
        render_mode: str | None = None
):
    if is_safety_gym_env(name):
        return make_safety_gym_env(name, max_steps, render_mode, make_sampler, ltl=True)
    else:
        return make_dmc_env(name, make_sampler, render_mode, ltl=True)


def make_sequence_env(
        name: str,
        max_steps: Optional[int] = None,  # TODO: add max_steps parameter to other environments
        render_mode: str | None = None
):
    if name.startswith('pretraining_'):
        underlying = name[len('pretraining_'):]
        underlying_env = make_sequence_env(underlying, max_steps, render_mode)
        propositions = get_env_attr(underlying_env, 'get_propositions')()
        impossible_assignments = get_env_attr(underlying_env, 'get_impossible_assignments')()
        return make_pretraining_env(propositions, impossible_assignments)
    elif is_safety_gym_env(name):
        return make_safety_gym_env(name, max_steps, render_mode)
    else:
        # return make_dmc_env(name, make_sampler, render_mode)
        raise NotImplementedError('DMC environments with sequences.')


def is_safety_gym_env(name: str) -> bool:
    return any([name.startswith(agent_name) for agent_name in ['Point', 'Car', 'Racecar', 'Doggo', 'Ant']])


def make_safety_gym_env(name: str, max_steps: Optional[int], render_mode: str | None = None,
                        ltl_sampler: Callable[[list[str]], LTLSampler] | None = None, ltl: bool = False):
    # noinspection PyUnresolvedReferences
    import safety_gymnasium
    from envs.zones.safety_gym_wrapper import SafetyGymWrapper
    from envs.seq_wrapper import SequenceWrapper

    if ltl:
        assert ltl_sampler is not None
    env = safety_gymnasium.make(name, render_mode=render_mode)
    env = SafetyGymWrapper(env)
    env = FlattenObservation(env)
    if ltl:
        propositions = get_env_attr(env, 'get_propositions')()
        env = LTLGoalWrapper(env, ltl_sampler(propositions))
    else:
        env = SequenceWrapper(env)
    env = TimeLimit(env, max_steps if max_steps else 1000)
    env = RemoveTruncWrapper(env)
    return env


def make_dmc_env(name: str, ltl_sampler: Callable[[list[str]], LTLSampler], render_mode: str | None = None):
    # noinspection PyUnresolvedReferences
    from dm_control import suite, viewer   # TODO: adapt to sequences
    import envs.dmc as dmc
    dmc.register_with_suite()
    from envs.dmc.dmc_gym_wrapper.dmc_gym_wrapper import DMCGymWrapper
    from envs.ldba_seq_wrapper import LDBAGraphWrapper

    env = suite.load(domain_name=name, task_name='ltl', visualize_reward=False)
    env = DMCGymWrapper(env, render_mode=render_mode)
    env = FlattenObservation(env)
    if name.startswith('ltl_cartpole'):
        # load alternate task
        env = DictWrapper(env)
        env = AlternateWrapper(env, ['yellow', 'green'])
        env = RemoveTruncWrapper(env)
    else:
        env = LTLGoalWrapper(env, ltl_sampler(get_env_attr(env, 'get_propositions')()))
        # env = GoalIndexWrapper(env, punish_termination=False)
        env = LDBAGraphWrapper(env, punish_termination=True)
        env = RemoveTruncWrapper(env)
    return env


def make_pretraining_env(
        propositions: set[str],
        impossible_assignments: set[FrozenAssignment],
):
    from envs.pretraining.pretraining_env import PretrainingEnv
    from envs.seq_wrapper import SequenceWrapper

    env = PretrainingEnv(propositions, impossible_assignments)
    env = SequenceWrapper(env)
    env = TimeLimit(env, max_episode_steps=100)
    env = RemoveTruncWrapper(env)
    return env
