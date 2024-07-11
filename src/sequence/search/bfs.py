from dataclasses import dataclass

from torch import nn

from ltl.automata import LDBA, LDBATransition, LDBASequence
from preprocessing import preprocessing


@dataclass(eq=True, frozen=True)
class SearchNode:
    ldba_state: int
    sequence: LDBASequence
    visited_states: set[int]


class BFS:
    def __init__(self, model: nn.Module, depth: int):
        self.model = model

    def __call__(self, ldba: LDBA, ldba_state: int, obs) -> LDBASequence:
        seqs = self.bfs(ldba, ldba_state, obs)
        seq = max(seqs, key=lambda s: self.get_value(s, obs))
        seq = self.augment_sequence(ldba, ldba_state, seq)  # TODO: augment or not?
        return seq

    def bfs(self, ldba: LDBA, ldba_state: int, obs) -> list[LDBASequence]:
        visited: set[int] = set()
        min_length = 0
        queue = [SearchNode(ldba_state, (), set())]
        sequences = []
        while queue:
            node = queue.pop(0)
            visited.add(node.ldba_state)
            avoid_transitions = self.collect_avoid_transitions(ldba, node.ldba_state, node.visited_states)
            avoid = [a.valid_assignments for a in avoid_transitions]
            avoid = set() if not avoid else set.union(*avoid)
            for t in ldba.state_to_transitions[node.ldba_state]:
                if t.target in visited:
                    continue
                if t.target == t.source and not t.accepting:
                    continue
                scc = ldba.state_to_scc[t.target]
                if scc.bottom and not scc.accepting:
                    continue
                new_sequence = node.sequence + ((frozenset(t.valid_assignments), frozenset(avoid)),)
                if min_length > 0 and len(new_sequence) > min_length:
                    assert all(len(n.sequence) >= min_length for n in queue)
                    break
                if scc.accepting:
                    assert all(len(s) == len(new_sequence) for s in sequences)
                    sequences.append(new_sequence)
                    min_length = len(new_sequence)
                    continue
                new_node = SearchNode(t.target, new_sequence, node.visited_states | {node.ldba_state})
                queue.append(new_node)
        return sequences

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

    def get_value(self, seq: LDBASequence, obs) -> float:
        obs['goal'] = seq
        if not (isinstance(obs, list) or isinstance(obs, tuple)):
            obs = [obs]
        preprocessed = preprocessing.preprocess_obss(obs)
        _, value = self.model(preprocessed)
        return value.item()

    def augment_sequence(self, ldba: LDBA, ldba_state: int, seq: LDBASequence) -> LDBASequence:
        augmented_path = []
        visited = set()
        state = ldba_state
        for reach, a in seq:
            visited.add(state)
            avoid = set()
            found = False
            for t in ldba.state_to_transitions[state]:
                if t.valid_assignments == reach:
                    state = t.target
                    found = True
                    continue
                if t.source == t.target:
                    continue
                scc = ldba.state_to_scc[t.target]
                if (scc.bottom and not scc.accepting) or self.only_non_accepting_loops(ldba, t.target, visited):
                    avoid.update(frozenset(t.valid_assignments))
            assert found
            assert a.issubset(avoid)
            augmented_path.append((reach, frozenset(avoid)))
        return tuple(augmented_path)

    def only_non_accepting_loops(self, ldba: LDBA, state: int, visited: set[int]) -> bool:
        if state in visited:
            return True
        stack = [state]
        marked = set()
        while stack:
            state = stack.pop()
            for t in ldba.state_to_transitions[state]:
                if t.target in marked:
                    continue
                scc = ldba.state_to_scc[t.target]
                if scc.bottom and not scc.accepting:
                    continue
                if t.target in visited:
                    continue
                if t.accepting:
                    return False
                stack.append(t.target)
            marked.add(state)
        visited.update(marked)
        return True
