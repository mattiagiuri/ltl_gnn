from typing import Any, SupportsFloat

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType


class LtlWrapperShaped(gymnasium.Wrapper):

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, gymnasium.spaces.Dict):
            raise ValueError('LTL wrapper requires dict observations')
        env.observation_space['ltl_state'] = spaces.Discrete(4)
        self.state = 0
        self.num_steps = 0
        self.last_lidar = -1

    def info_to_label_id(self, info):
        if 'magenta' in info['label']:
            return 0
        elif 'green' in info['label']:
            return 1
        elif 'yellow' in info['label']:
            return 2
        elif 'blue' in info['label']:
            return 3
        else:
            return -1

    def label_id_to_color(self, id):
        return {
            0: 'magenta',
            1: 'green',
            2: 'yellow',
            3: 'blue'
        }[id]


    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, _, terminated, truncated, info = super().step(action)
        self.num_steps += 1
        if terminated:
            reward = -1
        else:
            label_id = self.info_to_label_id(info)
            if label_id == self.state:
                terminated = True
                reward = 1
            else:
                color = self.label_id_to_color(self.state)
                lidar = obs[f'{color}_zones_lidar'].max()
                if self.last_lidar == -1:
                    reward = 0
                else:
                    reward = 10 * (lidar - self.last_lidar)
                self.last_lidar = lidar
        obs['ltl_state'] = self.state
        return obs, reward, terminated, truncated, info


    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.state = np.random.randint(0, 4)
        self.last_lidar = -1
        obs['ltl_state'] = self.state
        return obs, info
