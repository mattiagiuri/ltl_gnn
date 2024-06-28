from typing import Any, SupportsFloat

import gymnasium
from gymnasium.core import WrapperObsType, WrapperActType

from ltl.automata import LDBAGraph


class LDBAToSequenceWrapper(gymnasium.Wrapper):
    """
    Wrapper that converts an LDBA to a list of reach-avoid sequences, which are added to the observation space.
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, gymnasium.spaces.Dict):
            raise ValueError('LDBA to sequence wrapper requires dict observations')
        self.sequences = None
        self.prev_ldba_state = None

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        self.complete_observation(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.prev_ldba_state = None
        self.complete_observation(obs)
        return obs, info

    def complete_observation(self, obs: WrapperObsType):
        assert self.ldba_state is not None
        obs['changed'] = False
        if self.ldba_state != self.prev_ldba_state:
            self.prev_ldba_state = self.ldba_state
            ldba_graph = LDBAGraph.from_ldba(self.ldba, self.ldba_state)
            paths = ldba_graph.paths
            self.sequences = [p.to_sequence() for p in paths]
            obs['changed'] = True
        obs['sequences'] = self.sequences
