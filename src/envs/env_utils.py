from typing import Callable, Optional

import gymnasium
from gymnasium.wrappers import FlattenObservation, TimeLimit

from envs.remove_trunc_wrapper import RemoveTruncWrapper


def get_env_attr(env, attr: str):
    if hasattr(env, attr):
        return getattr(env, attr)
    if hasattr(env, 'env'):
        return getattr(env.unwrapped, attr)
    else:
        raise AttributeError(f'Attribute {attr} not found in env.')


def make_env(
        name: str,
        sampler: Callable[[list[str]], Callable],
        max_steps: Optional[int] = None,
        render_mode: str | None = None,
        eval_mode: bool = False,
        terminate_on_acceptance: bool = True
):
    from envs.pretraining.pretraining_env import PretrainingEnv
    from envs.seq_wrapper import SequenceWrapper
    from envs.partially_ordered_wrapper import PartiallyOrderedWrapper
    from envs.ldba_to_seq_wrapper import LDBAToSequenceWrapper
    from envs.ldba_wrapper import LDBAWrapper
    from envs.ltl_wrapper import LTLWrapper

    if name.startswith('pretraining_'):
        underlying = name[len('pretraining_'):]
        underlying_env = make_env(underlying, sampler, max_steps, render_mode)
        propositions = get_env_attr(underlying_env, 'get_propositions')()
        impossible_assignments = get_env_attr(underlying_env, 'get_impossible_assignments')()
        env = PretrainingEnv(propositions, impossible_assignments)
        max_steps = max_steps or 100
    elif is_safety_gym_env(name):
        env = make_safety_gym_env(name, render_mode)
        max_steps = max_steps or 1000
    elif name.startswith('Letter'):
        env = make_letter_env(name, render_mode)
        max_steps = max_steps or 75
    else:
        raise NotImplementedError('DMC environments not implemented yet.')

    propositions = get_env_attr(env, 'get_propositions')()
    sample_task = sampler(propositions)
    if eval_mode:
        # env = PartiallyOrderedWrapper(env, sample_task)
        env = LTLWrapper(env, sample_task)
        env = LDBAWrapper(env, terminate_on_acceptance)
        env = LDBAToSequenceWrapper(env)
    else:
        env = SequenceWrapper(env, sample_task, False)
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = RemoveTruncWrapper(env)
    return env


def is_safety_gym_env(name: str) -> bool:
    return any([name.startswith(agent_name) for agent_name in ['Point', 'Car', 'Racecar', 'Doggo', 'Ant']])


def make_safety_gym_env(name: str, render_mode: str | None = None):
    # noinspection PyUnresolvedReferences
    import safety_gymnasium
    from envs.zones.safety_gym_wrapper import SafetyGymWrapper

    env = safety_gymnasium.make(name, render_mode=render_mode)
    env = SafetyGymWrapper(env)
    env = FlattenObservation(env)
    return env


def make_letter_env(name: str, render_mode: str | None = None):
    import envs.letter_world

    env = gymnasium.make(name, render_mode=render_mode)
    return env

# def make_dmc_env(
#         name: str,
#         task_wrapper_cls: Type,
#         sample_task: Callable,
#         max_steps: Optional[int] = None,
#         render_mode: str | None = None
# ):
#     # noinspection PyUnresolvedReferences
#     from dm_control import suite, viewer  # TODO: adapt to sequences
#     import envs.dmc as dmc
#     dmc.register_with_suite()
#     from envs.dmc.dmc_gym_wrapper.dmc_gym_wrapper import DMCGymWrapper
#     from envs.ldba_seq_wrapper import LDBAGraphWrapper
#
#     env = suite.load(domain_name=name, task_name='ltl', visualize_reward=False)
#     env = DMCGymWrapper(env, render_mode=render_mode)
#     env = FlattenObservation(env)
#     if name.startswith('ltl_cartpole'):
#         # load alternate task
#         env = DictWrapper(env)
#         env = AlternateWrapper(env, ['yellow', 'green'])
#         env = RemoveTruncWrapper(env)
#     else:
#         env = LTLGoalWrapper(env, ltl_sampler(get_env_attr(env, 'get_propositions')()))
#         # env = GoalIndexWrapper(env, punish_termination=False)
#         env = LDBAGraphWrapper(env, punish_termination=True)
#         env = RemoveTruncWrapper(env)
#     return env
