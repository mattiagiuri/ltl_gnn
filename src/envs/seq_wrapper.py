import random
from typing import Any, SupportsFloat, Callable

import gymnasium
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType

from ltl.automata import LDBASequence, EPSILON
from ltl.logic import Assignment


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
        self.is_reach_and_stay = False  # TODO: implement in a cleaner way, this is very hacky
        self.reach_and_stay_target = 500  # was 30 (/ 60) before
        self.max_stay = 0
        self.reach_and_stay_seen = 0
        self.num_reached = 0
        self.propositions = set(env.get_propositions())
        self.partial_reward = partial_reward
        self.obs = None
        self.info = None

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if (action == EPSILON).all():
            obs, _, terminated, truncated, info = self.apply_epsilon_action()
        else:
            assert not (action == EPSILON).any()
            obs, _, terminated, truncated, info = super().step(action)
        reach, avoid = self.goal_seq[self.num_reached]
        reward = 0.
        active_props = info['propositions']
        assignment = Assignment({p: (p in active_props) for p in self.propositions}).to_frozen()
        if assignment in avoid:
            reward = -1.
            info['violation'] = True
            terminated = True
        elif reach != EPSILON and assignment in reach:
            if self.is_reach_and_stay and self.num_reached >= 1:
                self.reach_and_stay_seen += 1
                self.max_stay = max(self.max_stay, self.reach_and_stay_seen)
                if self.reach_and_stay_seen >= self.reach_and_stay_target:
                    terminated = True
            else:
                self.num_reached += 1
                terminated = self.num_reached >= len(self.goal_seq)
            if terminated:
                info['success'] = True
            if self.partial_reward:
                reward = 1. if terminated else 1 / (len(self.goal_seq) - self.num_reached + 1)
            else:
                reward = 1. if terminated else 0
        self.obs = obs
        self.info = info
        obs = self.complete_observation(obs, info)
        info['max_stay'] = self.max_stay
        return obs, reward, terminated, truncated, info

    def apply_epsilon_action(self):
        assert self.goal_seq[self.num_reached][0] == EPSILON
        self.num_reached += 1
        return self.obs, 0.0, False, False, self.info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.goal_seq = self.sample_sequence()
        self.is_reach_and_stay = len(self.goal_seq) > 1 and Assignment.zero_propositions(
            self.propositions).to_frozen() in self.goal_seq[1][1]
        # assert self.is_reach_and_stay
        self.reach_and_stay_seen = 0
        self.max_stay = 0
        self.num_reached = 0
        obs = self.complete_observation(obs, info)
        self.obs = obs
        self.info = info
        return obs, info

    def complete_observation(self, obs: WrapperObsType, info: dict[str, Any] = None) -> WrapperObsType:
        return {
            'features': obs,
            'goal': self.goal_seq[self.num_reached:],
            'initial_goal': self.goal_seq,
            'propositions': info['propositions'],
        }
