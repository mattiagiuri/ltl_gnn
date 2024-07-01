import copy
import math
from dataclasses import dataclass, field
from typing import Optional

from torch import nn

from ltl.automata import LDBA, LDBATransition
from ltl.automata.ldba_graph import Path
from ltl.logic import FrozenAssignment
from preprocessing import preprocessing
from utils import PriorityQueue


@dataclass(eq=True, frozen=True)
class SearchNode:
    ldba_state: int
    path: tuple[tuple[frozenset[FrozenAssignment], frozenset[FrozenAssignment]], ...]  # (valid assignments, avoid)
    prev: Optional['SearchNode'] = field(default=None, compare=False)

    def __lt__(self, other):
        return self.ldba_state > other.ldba_state  # arbitrary way to break ties


class LDBASequenceSearch:
    # CACHE: dict[tuple[str, int], Path] = {}

    def __init__(self, model: nn.Module, depth: int):
        self.model = model
        self.depth = depth

    def __call__(self, ldba: LDBA, ldba_state: int, visited_ldba_states: set[int], obs):
        # if (ldba.formula, current_state) in self.CACHE:
        #     return self.CACHE[(ldba.formula, current_state)]
        path = self.dijkstra(ldba, ldba_state, visited_ldba_states, copy.deepcopy(obs))
        # self.CACHE[(ldba.formula, current_state)] = path
        return path

    def dijkstra(self, ldba: LDBA, ldba_state: int, visited_ldba_states: set[int], obs):
        pq = PriorityQueue()
        pq.push(SearchNode(ldba_state, ()), 0)
        visited = set()
        found = False
        goal_node = None
        while not pq.is_empty():
            if found:
                break
            node, cost = pq.pop()
            visited.add(node.ldba_state)
            avoid_transitions = self.collect_avoid_transitions(ldba, node.ldba_state, visited_ldba_states)
            avoid_assignments = frozenset()
            if len(avoid_transitions) > 0:
                avoid_assignments = frozenset(set.union(*[avoid.valid_assignments for avoid in avoid_transitions]))
            for transition in ldba.state_to_transitions[node.ldba_state]:
                if transition.accepting and transition.source == transition.target:
                    found = True
                    goal_node = node
                    break
                if transition in avoid_transitions or transition.target in visited:
                    continue
                next_path = node.path + ((frozenset(transition.valid_assignments), avoid_assignments),)
                if len(next_path) > self.depth:
                    next_path = next_path[-self.depth:]
                new_cost = cost + self.compute_transition_cost(next_path, obs)
                next_node = SearchNode(transition.target, next_path, node)
                if next_node in pq:
                    if new_cost < pq[next_node]:
                        pq.change_priority(next_node, cost)
                else:
                    pq.push(next_node, new_cost)
        assert found and goal_node is not None
        path = []
        while goal_node.prev is not None:
            path.append(goal_node.path[-1])
            goal_node = goal_node.prev
        return self.to_sequence(reversed(path))

    @staticmethod
    def collect_avoid_transitions(ldba: LDBA, state: int, visited_ldba_states: set[int]) -> set[LDBATransition]:
        avoid = set()
        for transition in ldba.state_to_transitions[state]:
            if transition.source == transition.target:
                continue
            scc = ldba.state_to_scc[transition.target]
            if scc.bottom and not scc.accepting or transition.target in visited_ldba_states:
                avoid.add(transition)
        return avoid

    def compute_transition_cost(self, path: tuple[tuple[set[FrozenAssignment], set[FrozenAssignment]], ...], obs) -> float:
        if len(path) == 1:
            cost = -math.log(self.get_value(obs, path))
        else:
            cost = math.log(self.get_value(obs, path[:-1])) - math.log(self.get_value(obs, path))
        cost = max(0.0, cost)
        return cost

    def get_value(self, obs, path) -> float:
        obs['goal'] = self.to_sequence(path)
        if not (isinstance(obs, list) or isinstance(obs, tuple)):
            obs = [obs]
        preprocessed = preprocessing.preprocess_obss(obs)
        _, value = self.model(preprocessed)
        return max(1e-8, min(1.0, value.item()))

    @staticmethod
    def to_sequence(path):
        seq = []
        for reach, avoid in path:
            # if len(reach) == 13:  # TODO
            #     continue
            assert len(reach) == 1
            reach = list(reach)[0]
            positive = [t[0] for t in reach if t[1]]
            assert len(positive) == 1
            reach = positive[0]
            assert len(avoid) <= 1
            if len(avoid) == 1:
                avoid = list(avoid)[0]
                positive = [t[0] for t in avoid if t[1]]
                assert len(positive) == 1
                avoid = positive[0]
            else:
                avoid = 'empty'
            seq.append((reach, avoid))
        return seq
