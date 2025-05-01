from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, WrapperObsType
from gymnasium.spaces import Box

from ltl.logic import Assignment
from envs.zones.quadrants import Quadrant


class SafetyGymWrapper(gymnasium.Wrapper):
    """
    A wrapper from safety gymnasium LTL environments to the gymnasium API.
    """

    def __init__(self, env: Any, wall_sensor=True, agent_pos=True):
        super().__init__(env)
        self.render_parameters.camera_name = 'track'
        self.render_parameters.width = 256
        self.render_parameters.height = 256
        self.num_lidar_bins = env.unwrapped.task.lidar_conf.num_bins
        obs_keys = env.observation_space.spaces.keys()
        self.colors = set()
        for key in obs_keys:
            if key.endswith('zones_lidar'):
                color = key.split('_')[0]
                self.colors.add(color)

        self.areas = ['right', 'top']
        self.observation_space = spaces.Dict(env.observation_space)  # copy the observation space
        if wall_sensor:
            self.observation_space['wall_sensor'] = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float64)
        if agent_pos:
            self.observation_space['agent_pos'] = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)
        self.last_dist = None

    def step(self, action: ActType):
        obs, reward, cost, terminated, truncated, info = super().step(action)
        if 'wall_sensor' in info:
            obs['wall_sensor'] = info['wall_sensor']
        if 'agent_pos' in info:
            obs['agent_pos'] = info['agent_pos']

        quadrant_props = []
        x_agent, y_agent = info['agent_pos']

        if x_agent >= 0:
            quadrant_props.append('right')
        if y_agent >= 0:
            quadrant_props.append('top')

        info['propositions'] = {c for c in self.colors if info[f'cost_zones_{c}'] > 0} | set(quadrant_props)
        if 'cost_ltl_walls' in info:
            terminated = terminated or info['cost_ltl_walls'] > 0
            reward = -1. if info['cost_ltl_walls'] > 0 else 0.
        return obs, reward, terminated, truncated, info

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        info['propositions'] = []
        info['zone_quadrants'] = self.get_zone_quadrants()
        obs['wall_sensor'] = np.array([0, 0, 0, 0])
        obs['agent_pos'] = np.array([0, 0])
        return obs, info

    def get_propositions(self) -> list[str]:
        return list(sorted(self.colors)) + self.areas

    def get_possible_assignments(self) -> list[Assignment]:
        # return Assignment.zero_or_one_propositions(set(self.get_propositions()))
        props = set(self.get_propositions())
        return [
            Assignment.where('blue', propositions=props),
            Assignment.where('green', propositions=props),
            Assignment.where('magenta', propositions=props),
            Assignment.where('yellow', propositions=props),
            Assignment.where('right', propositions=props),
            Assignment.where('top', propositions=props),
            Assignment.where('blue', 'right', propositions=props),
            Assignment.where('green', 'right', propositions=props),
            Assignment.where('magenta', 'right', propositions=props),
            Assignment.where('yellow', 'right', propositions=props),
            Assignment.where('blue', 'top', propositions=props),
            Assignment.where('green', 'top', propositions=props),
            Assignment.where('magenta', 'top', propositions=props),
            Assignment.where('yellow', 'top', propositions=props),
            Assignment.where('right', 'top', propositions=props),
            Assignment.where('blue', 'right', 'top', propositions=props),
            Assignment.where('green', 'right', 'top', propositions=props),
            Assignment.where('magenta', 'right', 'top', propositions=props),
            Assignment.where('yellow', 'right', 'top', propositions=props),
            Assignment.zero_propositions(props),
        ]

    def get_zone_quadrants(self):
        zone_quadrants = {}
        for color in self.colors:
            quadrants = set()
            for idx in [0, 1]:
                x, y = self.zone_positions[f"{color}_zone{idx}"]
                if x <= 0 and y >= 0:
                    quadrants.add(Quadrant.TOP_LEFT)
                elif x >= 0 and y >= 0:
                    quadrants.add(Quadrant.TOP_RIGHT)
                elif x <= 0 and y <= 0:
                    quadrants.add(Quadrant.BOTTOM_LEFT)
                elif x >= 0 and y <= 0:
                    quadrants.add(Quadrant.BOTTOM_RIGHT)
            zone_quadrants[color] = quadrants
        return zone_quadrants
