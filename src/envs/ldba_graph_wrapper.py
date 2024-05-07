import functools
from typing import Any, SupportsFloat

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType

from ltl.automata import ltl2ldba, LDBA
from model.ltl import LDBAGraph


class LDBAGraphWrapper(gymnasium.Wrapper):
    """
    Wrapper that converts an LTL goal to a positive and negative LDBA graph, which are added to the observation space.
    """

    def __init__(self, env: gymnasium.Env, punish_termination=False):
        super().__init__(env)
        if not isinstance(env.observation_space, gymnasium.spaces.Dict):
            raise ValueError('Transition graph wrapper requires dict observations')
        if 'goal' not in env.observation_space.spaces:
            raise ValueError('Transition graph wrapper requires goal in observation space')
        propositions = env.get_wrapper_attr('get_propositions')()
        impossible_assignments = env.get_wrapper_attr('get_impossible_assignments')()
        num_features = 2 ** len(propositions) - len(impossible_assignments) + 2
        assert num_features == 7  # TODO
        self.observation_space['pos_graph'] = spaces.Graph(
            node_space=spaces.Box(-np.inf, np.inf, shape=(num_features,)),
            edge_space=None
        )
        self.punish_termination = punish_termination
        self.ldba = None
        self.ldba_state = None

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, _, terminated, truncated, info = super().step(action)
        self.ldba_state, accepting = self.ldba.get_next_state(self.ldba_state, info['propositions'])
        self.complete_observation(obs)
        reward = 0.
        if self.punish_termination and terminated:
            reward = -1.
        elif accepting:  # TODO: add check if it is a finite property (i.e. single state with accepting loop)
            reward = 1.
            terminated = True
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.ldba = self.construct_ldba(obs['goal'])
        self.ldba_state = self.ldba.initial_state
        self.complete_observation(obs)
        return obs, info

    def complete_observation(self, obs: WrapperObsType):
        pos_graph, neg_graph = LDBAGraph.from_ldba(self.ldba, self.ldba_state)
        obs['pos_graph'] = pos_graph

    @functools.cache
    def construct_ldba(self, formula: str) -> LDBA:
        propositions = frozenset(self.env.get_wrapper_attr('get_propositions')())
        ldba = ltl2ldba(formula, propositions, simplify_labels=False)
        assert ldba.check_valid()
        ldba.complete_sink_state()
        impossible_assignments = self.env.get_wrapper_attr('get_impossible_assignments')()
        ldba.prune_impossible_transitions(impossible_assignments)
        ldba.compute_sccs()
        initial_scc = ldba.state_to_scc[ldba.initial_state]
        if initial_scc.bottom and not initial_scc.accepting:
            raise ValueError(f'The language of the LDBA for {formula} is empty.')
        return ldba
