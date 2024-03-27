from math import prod
from typing import Any

import gymnasium
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, WrapperObsType
from gymnasium.spaces import Box


class SafetyGymWrapper(gymnasium.Wrapper):
    """
    A wrapper from safety gymnasium LTL environments to the gymnasium API.
    """

    def __init__(self, env: Any):
        super().__init__(env)
        self.render_parameters.camera_name = 'track'
        self.render_parameters.width = 256
        self.render_parameters.height = 256
        self.num_lidar_bins = env.unwrapped.task.lidar_conf.num_bins
        obs_keys = env.observation_space.spaces.keys()
        self.color_to_obs_index = {}
        obs_array_pos = 0
        for key in obs_keys:
            if key.endswith('zones_lidar'):
                color = key.split('_')[0]
                self.color_to_obs_index[color] = obs_array_pos
            obs_array_pos += prod(env.obs_space_dict.spaces[key].shape)
        self.colors = self.color_to_obs_index.keys()
        self.observation_space = spaces.Dict(env.observation_space)  # copy the observation space
        self.observation_space['wall_sensor'] = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float64)
        self.last_dist = None

    def step(self, action: ActType):
        obs, reward, cost, terminated, truncated, info = super().step(action)
        obs['wall_sensor'] = info['wall_sensor']
        self.calculate_labels(info)
        terminated = terminated or 'wall' in info['label']
        return obs, reward, terminated, truncated, info

    def calculate_labels(self, info):
        info['label'] = []
        for color in self.colors:
            if info[f'cost_zones_{color}'] > 0:
                info['label'].append(color)
        if info['cost_ltl_walls'] > 0:
            info['label'].append('wall')

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        info['label'] = []
        obs['wall_sensor'] = np.array([0, 0, 0, 0])
        return obs, info

    @staticmethod
    def get_propositions():
        return ['magenta', 'green', 'yellow', 'blue']
