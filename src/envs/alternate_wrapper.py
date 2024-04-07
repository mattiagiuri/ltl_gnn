import random
from typing import Any, SupportsFloat

import gymnasium
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType


class AlternateWrapper(gymnasium.Wrapper):
    """
    Wrapper for a task that requires alternating infinitely often between two different propositions.
    """

    def __init__(self, env: gymnasium.Env, propositions: list[str]):
        super().__init__(env)
        if not isinstance(env.observation_space, gymnasium.spaces.Dict):
            raise ValueError('Alternation wrapper requires dict observations')
        if len(propositions) != 2:
            raise ValueError('Alternation wrapper requires exactly two propositions')
        self.observation_space['goal_index'] = spaces.Discrete(len(propositions))
        self.proposition_to_index = {p: i for i, p in enumerate(propositions)}
        self.goal = None

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, _, terminated, truncated, info = super().step(action)
        reward = 0.
        if terminated:
            reward = -1.
        elif self.goal in info['propositions']:
            reward = 1.
            self.goal = [p for p in self.proposition_to_index if p != self.goal][0]
        obs['goal_index'] = self.proposition_to_index[self.goal]
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.goal = random.choice(list(self.proposition_to_index.keys()))
        obs['goal_index'] = self.proposition_to_index[self.goal]
        return obs, info
