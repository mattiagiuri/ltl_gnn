import functools
from typing import Any, SupportsFloat

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType

from envs import get_env_attr
from ltl.automata import ltl2ldba, LDBA
from ltl.logic import Assignment
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
        propositions = get_env_attr(env, 'get_propositions')()
        impossible_assignments = get_env_attr(env, 'get_impossible_assignments')()
        num_features = 2 ** len(propositions) - len(impossible_assignments) + 2
        self.observation_space['pos_graph'] = spaces.Graph(
            node_space=spaces.Box(-np.inf, np.inf, shape=(num_features,)),
            edge_space=None
        )
        self.observation_space['neg_graph'] = spaces.Graph(
            node_space=spaces.Box(-np.inf, np.inf, shape=(num_features,)),
            edge_space=None
        )
        self.punish_termination = punish_termination
        self.ldba = None
        self.ldba_state = None
        self.num_success = 0

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, _, terminated, truncated, info = super().step(action)
        pos_graph = LDBAGraph.from_ldba(self.ldba, self.ldba_state)[0]
        root_assignments = pos_graph.root_assignments
        self.ldba_state, accepting = self.ldba.get_next_state(self.ldba_state, info['propositions'])
        self.complete_observation(obs)
        reward = 0.
        if self.punish_termination and terminated:
            reward = -1.
        elif accepting:  # TODO: add check if it is a finite property (i.e. single state with accepting loop)
            reward = 1.
            terminated = True  # TODO: properly handle this depending on the task (omega regular or not)
        else:
            scc = self.ldba.state_to_scc[self.ldba_state]
            if scc.bottom and not scc.accepting:
                reward = -1.  # TODO
                terminated = True
            assignment = Assignment({p: (p in info['propositions']) for p in self.ldba.propositions}).to_frozen()
            if assignment in root_assignments:
                reward = 1.  # TODO
                self.num_success += 1
                self.complete_observation(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.ldba = self.construct_ldba(obs['goal'])
        self.ldba_state = self.ldba.initial_state
        self.num_success = 0
        self.complete_observation(obs)
        return obs, info

    def complete_observation(self, obs: WrapperObsType):
        pos_graph, neg_graph = LDBAGraph.from_ldba(self.ldba, self.ldba_state)
        obs['pos_graph'] = pos_graph
        obs['neg_graph'] = neg_graph
        seq = self.goal_to_seq(obs['goal'])
        if self.num_success > 0:
            seq = seq[:-self.num_success]
        obs['seq'] = seq

    def goal_to_seq(self, goal: str) -> list[str]:
        return list(reversed(goal
                             .replace('(', '')
                             .replace(')', '')
                             .replace('F', '')
                             .replace('&', '')
                             .strip()
                             .split()))

    @functools.cache
    def construct_ldba(self, formula: str) -> LDBA:
        propositions = get_env_attr(self.env, 'get_propositions')()
        ldba = ltl2ldba(formula, propositions, simplify_labels=False)
        assert ldba.check_valid()
        ldba.complete_sink_state()
        impossible_assignments = get_env_attr(self.env, 'get_impossible_assignments')()
        ldba.prune_impossible_transitions(impossible_assignments)
        ldba.compute_sccs()
        initial_scc = ldba.state_to_scc[ldba.initial_state]
        if initial_scc.bottom and not initial_scc.accepting:
            raise ValueError(f'The language of the LDBA for {formula} is empty.')
        return ldba
