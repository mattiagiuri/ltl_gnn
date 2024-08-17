from dataclasses import dataclass
from typing import Optional

from torch import nn

from ltl.automata import LDBA, LDBATransition, LDBASequence, EPSILON
from sequence.search import SequenceSearch


@dataclass
class Path:
    reach_avoid: list[tuple[LDBATransition, set[LDBATransition]]]
    loop_index: int

    def __len__(self):
        return len(self.reach_avoid)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Path(self.reach_avoid[item], self.loop_index)
        return self.reach_avoid[item]

    def __str__(self):
        p = [(reach.source, {a.target for a in avoid}) for reach, avoid in self[:self.loop_index]]
        loop = [(reach.source, {a.target for a in avoid}) for reach, avoid in self[self.loop_index:]]
        return str(p + loop * 3)

    def prepend(self, reach: LDBATransition, avoid: set[LDBATransition]) -> 'Path':
        return Path([(reach, avoid)] + self.reach_avoid, self.loop_index)

    def to_sequence(self, num_loops: int) -> LDBASequence:
        seq = [self.reach_avoid_to_assignments(r, a) for r, a in self.reach_avoid[:self.loop_index]]
        loop = [self.reach_avoid_to_assignments(r, a) for r, a in self.reach_avoid[self.loop_index:]]
        seq = seq + loop * num_loops
        return tuple(seq)

    @staticmethod
    def reach_avoid_to_assignments(reach: LDBATransition, avoid: set[LDBATransition]) -> tuple[frozenset, frozenset]:
        avoid = [a.valid_assignments for a in avoid]
        avoid = set() if not avoid else set.union(*avoid)
        reach = EPSILON if reach.is_epsilon() else frozenset(reach.valid_assignments)
        return reach, frozenset(avoid)


class ExhaustiveSearch(SequenceSearch):
    def __init__(self, model: nn.Module, propositions, num_loops: int):
        super().__init__(model, propositions)
        self.num_loops = num_loops

    def __call__(self, ldba: LDBA, ldba_state: int, obs) -> LDBASequence:
        seqs = ExhaustiveSearch.all_sequences(ldba, ldba_state, self.num_loops)
        return max(seqs, key=lambda s: self.get_value(s, obs))

    @staticmethod
    def all_sequences(ldba: LDBA, ldba_state: int, num_loops=1) -> list[LDBASequence]:
        paths = ExhaustiveSearch.dfs(ldba, ldba_state, [], {}, None)
        num_loops = 0 if ldba.is_finite_specification() else num_loops
        return [path.to_sequence(num_loops) for path in paths]

    @staticmethod
    def dfs(ldba: LDBA, state: int, current_path: list[LDBATransition], state_to_path_index: dict[int, int],
            accepting_transition: Optional[LDBATransition]) -> list[Path]:
        """
        Performs a depth-first search on the LDBA to find all simple paths leading to an accepting loop.
        Returns the list of simple paths, and a set of negative transitions that either (i) lead to a sink state,
        or (ii) only lead to non-accepting cycles.
        """
        state_to_path_index[state] = len(current_path)
        neg_transitions = set()
        paths = []
        for transition in ldba.state_to_transitions[state]:
            scc = ldba.state_to_scc[transition.target]
            if scc.bottom and not scc.accepting:
                neg_transitions.add(transition)
            else:
                current_path.append(transition)
                stays_in_scc = scc == ldba.state_to_scc[transition.source]
                updated_accepting_transition = accepting_transition
                if transition.accepting and stays_in_scc:
                    updated_accepting_transition = transition
                if transition.target in state_to_path_index:  # found cycle
                    if updated_accepting_transition in current_path[state_to_path_index[transition.target]:]:
                        # found accepting cycle
                        path = Path(reach_avoid=[], loop_index=state_to_path_index[transition.target])
                        future_paths = [path]
                    else:
                        # found non-accepting cycle
                        current_path.pop()
                        if transition.source != transition.target:
                            neg_transitions.add(transition)
                        continue
                else:
                    future_paths = ExhaustiveSearch.dfs(ldba, transition.target, current_path, state_to_path_index,
                                                        updated_accepting_transition)
                    if len(future_paths) == 0:
                        neg_transitions.add(transition)
                for fp in future_paths:
                    # avoid transitions can only be added once the recursion is finished, so only set() for now
                    paths.append(fp.prepend(transition, set()))
                current_path.pop()

        del state_to_path_index[state]
        paths = ExhaustiveSearch.prune_paths(paths)
        for path in paths:
            path[0][1].update(neg_transitions)  # now we update the negative transitions
        return paths

    @staticmethod
    def prune_paths(paths: list[Path]) -> list[Path]:
        to_remove = set()
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if i in to_remove or j in to_remove:
                    continue
                if len(paths[i]) < len(paths[j]):
                    if ExhaustiveSearch.check_path_contained(paths[j], paths[i]):
                        to_remove.add(j)
                elif len(paths[i]) > len(paths[j]):
                    if ExhaustiveSearch.check_path_contained(paths[i], paths[j]):
                        to_remove.add(i)
                if i in to_remove:
                    break
        paths = [paths[i] for i in range(len(paths)) if i not in to_remove]
        return paths

    @staticmethod
    def check_path_contained(path1: Path, path2: Path) -> bool:
        assert len(path2) < len(path1)
        path1 = [t[0].valid_assignments for t in path1]
        path2 = [t[0].valid_assignments for t in path2]
        acc_pos = 0
        found = False
        for p in path1:
            if p.issubset(path2[acc_pos]):
                acc_pos += 1
                if acc_pos == len(path2):
                    found = True
                    break
        return found
