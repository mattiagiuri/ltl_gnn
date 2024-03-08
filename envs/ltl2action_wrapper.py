from typing import Any, SupportsFloat

import gymnasium
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType

from envs.ltl_wrapper import LtlWrapper


class Ltl2ActionWrapper(gymnasium.Wrapper):

    def __init__(self, env: LtlWrapper):
        super().__init__(env)
        self.reduced_observation_space = spaces.Dict({k: v for k, v in env.observation_space.items() if k != 'ltl_state'})
        self.observation_space = spaces.flatten_space(self.reduced_observation_space)

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        del obs['ltl_state']
        obs = spaces.flatten(self.reduced_observation_space, obs)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        del obs['ltl_state']
        return spaces.flatten(self.reduced_observation_space, obs), info

    def sample_ltl_goal(self):
        event = self.label_to_event(self.ltl_state_to_label(self.env.state))
        return 'eventually', event

    def get_events(self, info):
        labels = info['label']
        return ''.join([self.label_to_event(label) for label in labels])

    @staticmethod
    def label_to_event(label: str) -> str:
        return label[0]

    @staticmethod
    def ltl_state_to_label(ltl_state: int) -> str:
        return {
            0: 'magenta',
            1: 'green',
            2: 'yellow',
            3: 'blue'
        }[ltl_state]

    @staticmethod
    def get_propositions():
        return ['m', 'g', 'y', 'b']
