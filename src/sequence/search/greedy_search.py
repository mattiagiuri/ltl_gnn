from dataclasses import dataclass

from torch import nn

from ltl.automata import LDBA, LDBATransition, LDBASequence
from ltl.logic import FrozenAssignment
from preprocessing import preprocessing


@dataclass(eq=True, frozen=True)
class Path:
    sequence: LDBASequence  # (valid assignments, avoid)
    ldba_states: tuple[int, ...]
    accepting: tuple[bool, ...]

    def extend(self, reach: set[FrozenAssignment], avoid: set[FrozenAssignment], state: int, accepting: bool) -> 'Path':
        path = self.sequence + ((frozenset(reach), frozenset(avoid)),)
        states = self.ldba_states + (state,)
        acc = self.accepting + (accepting,)
        return Path(path, states, acc)

    def __repr__(self):
        return self.sequence.__repr__()


@dataclass(eq=True, frozen=True)
class SearchNode:
    ldba_state: int
    path: Path


class GreedySearch:
    def __init__(self, model: nn.Module, depth: int):
        self.model = model
        self.depth = depth

    def __call__(self, ldba: LDBA, ldba_state: int, obs) -> LDBASequence:
        path = self.search(ldba, ldba_state, obs)
        path = self.augment_path(ldba, path)
        return path.sequence

    def search(self, ldba: LDBA, ldba_state: int, obs) -> Path:
        states_on_path: set[int] = set()
        accepting_paths: set[Path] = set()

        def dfs(state: int, path: Path, depth: int, states_on_current_path: set[int]) -> set[Path]:
            if len(path.accepting) > 0 and path.accepting[-1]:
                accepting_paths.add(path)
                return {path}
            if depth >= self.depth:
                return {path}
            avoid = self.collect_avoid_transitions(ldba, state,
                                                   set())  # TODO: replace with states_on_current_path | states_on_path
            avoid_assignments = [a.valid_assignments for a in avoid]
            avoid = set() if not avoid_assignments else set.union(*avoid_assignments)
            paths = set()
            new_states_on_current_path = states_on_current_path | {state}
            for t in ldba.state_to_transitions[state]:
                if t.target in states_on_current_path or t.target in states_on_path:
                    continue
                if t.target == t.source and not t.accepting:
                    continue
                scc = ldba.state_to_scc[t.target]
                if scc.bottom and not scc.accepting:
                    continue
                new_path = path.extend(t.valid_assignments, avoid, t.target, t.accepting)
                paths.update(dfs(t.target, new_path, depth + 1, new_states_on_current_path))

            # TODO: implement pruning!!!

            return paths

        path = Path((), (ldba_state,), ())  # TODO: implement backtracking
        states_on_path.add(ldba_state)
        index = 0
        while len(path.accepting) == 0 or not path.accepting[-1]:
            paths = dfs(path.ldba_states[-1], path, 0, set())
            path_values = {p: self.get_value(p, obs) for p in paths}
            best_path, value = max(path_values.items(), key=lambda x: x[1])
            if len(accepting_paths) > 0:
                accepting_path_values = {p: self.get_value(p, obs) for p in accepting_paths}
                best_accepting_path, accepting_value = max(accepting_path_values.items(), key=lambda x: x[1])
                if accepting_value > value:
                    path = best_accepting_path
                    continue
            next_state = best_path.ldba_states[index + 1]
            path = path.extend(*best_path.sequence[index], next_state, best_path.accepting[index])
            states_on_path.add(next_state)
            index += 1
        return path

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

    def get_value(self, path: Path, obs) -> float:
        obs['goal'] = path.sequence
        if not (isinstance(obs, list) or isinstance(obs, tuple)):
            obs = [obs]
        preprocessed = preprocessing.preprocess_obss(obs)
        _, value = self.model(preprocessed)
        return value.item()

    def augment_path(self, ldba: LDBA, path: Path):
        augmented_path = []
        visited = set()
        for i, state in enumerate(path.ldba_states[:-1]):
            avoid = set()
            visited.add(state)
            for t in ldba.state_to_transitions[state]:
                if t.valid_assignments == path.sequence[i][0]:
                    continue
                if t.source == t.target:
                    continue
                scc = ldba.state_to_scc[t.target]
                if (scc.bottom and not scc.accepting) or self.only_non_accepting_loops(ldba, t.target, visited):
                    if len(avoid) >= 1:
                        continue
                    avoid.update(frozenset(t.valid_assignments))
                    continue
            augmented_path.append((path.sequence[i][0], frozenset(avoid)))
        return Path(tuple(augmented_path), path.ldba_states, path.accepting)

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
