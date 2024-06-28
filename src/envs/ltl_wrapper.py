from typing import Any, SupportsFloat, Callable

import gymnasium
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType


class LTLWrapper(gymnasium.Wrapper):
    LTL_CHARSET = 'abcdefghijklmnopqrstuvwxyzFG!()&|'

    def __init__(self, env: gymnasium.Env, sample_task: Callable[[], str]):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            'features': env.observation_space,
            'goal': spaces.Text(max_length=100, charset=self.LTL_CHARSET)
        })
        self.sample_task = sample_task
        self.goal = None

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        obs = {'features': obs, 'goal': self.goal}
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.goal = self.sample_task()
        obs = {'features': obs, 'goal': self.goal}
        return obs, info
