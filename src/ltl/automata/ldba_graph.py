from dataclasses import dataclass, field
from typing import Optional

from ltl.automata import LDBA, LDBATransition
from ltl.logic import FrozenAssignment


@dataclass
class Node:
    id: Optional[int] = None
    positive: set[FrozenAssignment] = field(default_factory=set)
    negative: set[FrozenAssignment] = field(default_factory=set)
    pos_label: str = ''
    neg_label: str = ''
    neighbors: set[int] = field(default_factory=set)


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

    def to_sequence(self, max_length: Optional[int] = None) -> list[tuple[str, str]]:
        # TODO: implement max_length, account for loops and finite-horizon tasks
        # length = len(self.reach_avoid) if max_length is None else max_length
        seq = []
        for reach, avoid in self.reach_avoid:
            if len(reach._valid_assignments) == 13:
                continue
            assert len(reach._valid_assignments) == 1
            reach = reach.positive_label
            assert len(avoid) <= 1
            if len(avoid) == 1:
                avoid = list(avoid)[0]
                assert len(avoid._valid_assignments) == 1
                avoid_label = avoid.positive_label
            else:
                avoid_label = 'empty'
            seq.append((reach, avoid_label))
        return seq


class LDBAGraph:
    CACHE: dict[tuple[str, int], 'LDBAGraph'] = {}

    def __init__(self):
        self.nodes: list[Node] = []
        self.root_nodes: set[int] = set()
        self.paths: list[Path] = []

    def add_node(self, node: Node):
        node.id = len(self.nodes)
        self.nodes.append(node)

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def edges(self):
        return [(node.id, neighbor) for node in self.nodes for neighbor in node.neighbors]

    @classmethod
    def from_ldba(cls, ldba: LDBA, current_state: int) -> 'LDBAGraph':
        if not ldba.complete:
            raise ValueError('The LDBA must be complete. Make sure to call '
                             '`ldba.complete_sink_state()` before constructing the graph.')
        if not ldba.state_to_scc:
            raise ValueError('The SCCs of the LDBA must be initialised. Make sure to call '
                             '`ldba.compute_sccs()` before constructing the graph.')
        assert ldba.formula is not None
        if (ldba.formula, current_state) in cls.CACHE:
            return cls.CACHE[(ldba.formula, current_state)]
        graph = cls.construct_graph(ldba, current_state)
        cls.CACHE[(ldba.formula, current_state)] = graph
        return graph

    @classmethod
    def construct_graph(cls, ldba: LDBA, current_state: int) -> 'LDBAGraph':
        paths: list[Path] = cls.dfs(ldba, current_state, [], {}, None)

        transition_to_node: dict[tuple[LDBATransition, frozenset[LDBATransition]], Node] = {}
        graph = LDBAGraph()

        def build_graph(remaining_paths: list[Path]) -> list[Node]:
            if len(remaining_paths) == 0:
                return []
            partition = {}
            for path in remaining_paths:
                s = (path[0][0], frozenset(path[0][1]))
                if s not in partition:
                    partition[s] = []
                if len(path) > 1:
                    partition[s].append(path[1:])
            nodes = []
            for transition, future_paths in partition.items():
                transition = (transition[0], frozenset(transition[1]))
                children = build_graph(future_paths)
                if transition in transition_to_node:
                    node = transition_to_node[transition]
                else:
                    node = Node(
                        positive=transition[0].valid_assignments,
                        pos_label=transition[0].positive_label
                    )
                    neg = transition[1]
                    if neg:
                        node.negative = set.union(*[t.valid_assignments for t in neg]),
                        node.neg_label = ' OR '.join(t.positive_label for t in neg)
                    transition_to_node[transition] = node
                    graph.add_node(node)
                for child in children:
                    node.neighbors.add(child.id)
                nodes.append(node)
            return nodes

        roots = build_graph(paths)
        graph.root_nodes = set(node.id for node in roots)
        graph.paths = paths
        # add loops
        for path in paths:
            final_node = transition_to_node[(path[-1][0], frozenset(path[-1][1]))]
            loop_node = transition_to_node[(path[path.loop_index][0], frozenset(path[path.loop_index][1]))]
            final_node.neighbors.add(loop_node.id)
        return graph

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
                    future_paths = LDBAGraph.dfs(ldba, transition.target, current_path, state_to_path_index,
                                                 updated_accepting_transition)
                    if len(future_paths) == 0:
                        neg_transitions.add(transition)
                for fp in future_paths:
                    # avoid transitions can only be added once the recursion is finished, so only set() for now
                    paths.append(fp.prepend(transition, set()))
                current_path.pop()

        del state_to_path_index[state]
        paths = LDBAGraph.prune_paths(paths)
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
                    if LDBAGraph.check_path_contained(paths[j], paths[i]):
                        to_remove.add(j)
                elif len(paths[i]) > len(paths[j]):
                    if LDBAGraph.check_path_contained(paths[i], paths[j]):
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
