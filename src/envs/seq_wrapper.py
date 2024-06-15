import random
from typing import Any, SupportsFloat, Callable

import gymnasium
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType

from envs import get_env_attr
from sequence import RandomSequenceSampler
from sequence.curriculum_sequence_sampler import CurriculumSequenceSampler
from sequence.fixed_sequence_sampler import FixedSequenceSampler


class SequenceWrapper(gymnasium.Wrapper):
    """
    Wrapper that adds a reach-avoid sequence of propositions to the observation space.
    """

    def __init__(self, env: gymnasium.Env, sample_sequence: Callable[[], list[tuple[str, str]]], eval_mode=False):
        super().__init__(env)
        propositions = get_env_attr(env, 'get_propositions')()
        self.observation_space = spaces.Dict({
            'features': env.observation_space,
            'goal': spaces.Tuple((spaces.Text(max_length=100, charset=propositions),
                                 spaces.Text(max_length=100, charset=propositions)))
        })
        self.sample_sequence = sample_sequence
        self.goal_seq = None
        self.num_reached = 0
        self.eval_mode = eval_mode
        self.has_reached_prev_avoid = False

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, _, terminated, truncated, info = super().step(action)
        if self.eval_mode:
            assert len(self.goal_seq) == 2
        reach, avoid = self.goal_seq[self.num_reached]
        reward = 0.
        if self.eval_mode and self.num_reached == 1 and self.goal_seq[0][1] in info['propositions']:
            self.has_reached_prev_avoid = True
        if avoid in info['propositions']:
            if self.eval_mode and self.num_reached == 1 and not self.has_reached_prev_avoid:
                self.num_reached = 0
            else:
                reward = -1.
                info['violation'] = True
                terminated = True
        elif reach in info['propositions']:
            self.num_reached += 1
            terminated = self.num_reached >= len(self.goal_seq)
            if terminated:
                info['success'] = True
            # reward = 1. if terminated else 1 / (len(self.goal_seq) - self.num_reached + 1)
            reward = 1. if terminated else 0
        obs = self.complete_observation(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.goal_seq = self.sample_sequence()
        self.num_reached = 0
        self.has_reached_prev_avoid = False
        obs = self.complete_observation(obs)
        return obs, info

    def complete_observation(self, obs: WrapperObsType):
        return {
            'features': obs,
            'goal': self.goal_seq[self.num_reached:],
            'initial_goal': self.goal_seq
        }
