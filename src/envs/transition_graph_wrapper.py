import functools
import sys
from typing import Any, SupportsFloat

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType

from ltl.automata import ltl2ldba, LDBA
from model.ltl import TransitionGraph


class TransitionGraphWrapper(gymnasium.Wrapper):
    """
    Wrapper that converts an LTL goal to a transition graph, which is added to the observation space along with the
    currently active transitions.
    """

    def __init__(self, env: gymnasium.Env, punish_termination=False):
        super().__init__(env)
        if not isinstance(env.observation_space, gymnasium.spaces.Dict):
            raise ValueError('Transition graph wrapper requires dict observations')
        if 'goal' not in env.observation_space.spaces:
            raise ValueError('Transition graph wrapper requires goal in observation space')
        propositions = env.get_wrapper_attr('get_propositions')()
        num_features = len(propositions) + 3
        self.observation_space['transition_graph'] = spaces.Graph(
            node_space=spaces.Box(-np.inf, np.inf, shape=(num_features,)),
            edge_space=None
        )
        self.observation_space['active_transitions'] = spaces.Sequence(spaces.Discrete(sys.maxsize))
        self.punish_termination = punish_termination
        self.ldba = None
        self.ldba_state = None
        self.tg = None

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
        self.tg = TransitionGraph.from_ldba(self.ldba)
        self.complete_observation(obs)
        return obs, info

    def complete_observation(self, obs: WrapperObsType):
        obs['transition_graph'] = self.tg
        #obs['active_transitions'] = [self.tg.transition_to_index[t] for t in
        #                             self.ldba.state_to_transitions[self.ldba_state]]
        obs['active_transitions'] = [0]

    @functools.cache
    def construct_ldba(self, formula: str) -> LDBA:
        propositions = frozenset(self.env.get_wrapper_attr('get_propositions')())
        ldba = ltl2ldba(formula, propositions, simplify_labels=False)
        assert ldba.check_valid()
        ldba.complete_sink_state()
        impossible_assignments = self.env.get_wrapper_attr('get_impossible_assignments')()
        ldba.prune_impossible_transitions(impossible_assignments)
        return ldba
