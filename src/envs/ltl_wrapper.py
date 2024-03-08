from typing import Any, SupportsFloat

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType


class LtlWrapper(gymnasium.Wrapper):

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, gymnasium.spaces.Dict):
            raise ValueError('LTL wrapper requires dict observations')
        env.observation_space['ltl_state'] = spaces.Discrete(4)
        self.state = 0
        self.num_steps = 0

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, _, terminated, truncated, info = super().step(action)
        self.num_steps += 1
        def rew(labels, state):
            if 'magenta' in labels and state == 0:
                return 1., state
            elif 'green' in labels and state == 1:
                return 1., state
            elif 'yellow' in labels and state == 2:
                return 1., state
            elif 'blue' in labels and state == 3:
                return 1., state
            if terminated:
                return -1., state
            return 0, state
        reward, self.state = rew(info['label'], self.state)
        if reward > 0:
            terminated = True
        obs['ltl_state'] = self.state
        return obs, reward, terminated, truncated, info


    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.state = (self.state + 1) % 4
        obs['ltl_state'] = self.state
        return obs, info
