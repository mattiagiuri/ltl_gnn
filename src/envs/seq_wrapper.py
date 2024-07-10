import random
from typing import Any, SupportsFloat, Callable

import gymnasium
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType

from envs import get_env_attr
from ltl.automata import LDBASequence
from ltl.logic import Assignment
from sequence import RandomSequenceSampler
from sequence.curriculum_sequence_sampler import CurriculumSequenceSampler
from sequence.fixed_sequence_sampler import FixedSequenceSampler


class SequenceWrapper(gymnasium.Wrapper):
    """
    Wrapper that adds a reach-avoid sequence of propositions to the observation space.
    """

    def __init__(self, env: gymnasium.Env, sample_sequence: Callable[[], LDBASequence], partial_reward=False):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            'features': env.observation_space,
        })
        self.sample_sequence = sample_sequence
        self.goal_seq = None
        self.num_reached = 0
        self.propositions = set(env.get_propositions())
        self.partial_reward = False

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, _, terminated, truncated, info = super().step(action)
        reach, avoid = self.goal_seq[self.num_reached]
        reward = 0.
        active_props = info['propositions']
        assignment = Assignment({p: (p in active_props) for p in self.propositions}).to_frozen()
        if assignment in avoid:
            reward = -1.
            info['violation'] = True
            terminated = True
        elif assignment in reach:
            self.num_reached += 1
            terminated = self.num_reached >= len(self.goal_seq)
            if terminated:
                info['success'] = True
            if self.partial_reward:
                reward = 1. if terminated else 1 / (len(self.goal_seq) - self.num_reached + 1)
            else:
                reward = 1. if terminated else 0
        obs = self.complete_observation(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.goal_seq = self.sample_sequence()
        self.num_reached = 0
        obs = self.complete_observation(obs)
        return obs, info

    def complete_observation(self, obs: WrapperObsType):
        return {
            'features': obs,
            'goal': self.goal_seq[self.num_reached:],
            'initial_goal': self.goal_seq
        }
