import functools
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from ltl.logic.assignment import FrozenAssignment, Assignment
from ltl.logic.sympy_utils import to_sympy, simplify, to_str


class LDBA:
    def __init__(self, propositions: set[str]):
        self.propositions: tuple[str, ...] = tuple(sorted(propositions))
        self.num_states = 0
        self.num_transitions = 0
        self.initial_state = None
        self.state_to_transitions: dict[int, list[LDBATransition]] = {}
        self.state_to_incoming_transitions: dict[int, list[LDBATransition]] = {}
        self.sink_state: Optional[int] = None
        self.complete = False

    def add_state(self, state: int, initial=False):
        if state < 0:
            raise ValueError('State must be a positive integer.')
        if initial:
            if self.initial_state is not None:
                raise ValueError('Initial state already set.')
            self.initial_state = state
        self.num_states = max(self.num_states, state + 1)
        if state not in self.state_to_transitions:
            self.state_to_transitions[state] = []
        if state not in self.state_to_incoming_transitions:
            self.state_to_incoming_transitions[state] = []

    def contains_state(self, state: int) -> bool:
        return state <= self.num_states

    def add_transition(self, source: int, target: int, label: Optional[str], accepting: bool, simplify_label=True):
        if source < 0 or source >= self.num_states:
            raise ValueError('Source state must be a valid state index.')
        if target < 0 or target >= self.num_states:
            raise ValueError('Target state must be a valid state index.')
        if simplify_label and label is not None:
            label = to_str(simplify(to_sympy(label)))
        transition = LDBATransition(source, target, label, accepting, self.propositions)
        self.num_transitions += 1
        self.state_to_transitions[source].append(transition)
        self.state_to_incoming_transitions[target].append(transition)

    def check_valid(self) -> bool:
        """Checks that the LDBA satisfies the following conditions:
           - It has a deterministic first component
           - It has a deterministic second component
           - All transitions from the first to the second component are epsilon transitions
           - There are no other epsilon transitions
           - All transitions from the second component stay in the second component
           - All accepting transitions are in the second component
           - The first component may be empty
           - The LDBA is fully connected
        """
        if self.initial_state is None:
            return False
        first_visited = set()
        first_queue = [self.initial_state]
        second_states = set()
        found_accepting = False
        while first_queue:
            state = first_queue.pop(0)
            first_visited.add(state)
            if not self.check_deterministic_transitions(state):
                return False
            for transition in self.state_to_transitions[state]:
                if transition.is_epsilon():
                    if transition.target in first_visited:
                        return False  # epsilon transition in the first component
                    second_states.add(transition.target)
                else:
                    if transition.target in second_states:
                        return False  # transition from first to second component is not epsilon
                    if transition.target not in first_visited:
                        first_queue.append(transition.target)
                if transition.accepting:
                    found_accepting = True
        if found_accepting and len(second_states) > 0:
            return False  # accepting transition in the first component
        second_queue = list(second_states)
        second_visited = set()
        while second_queue:
            state = second_queue.pop(0)
            second_visited.add(state)
            if not self.check_deterministic_transitions(state):
                return False
            for transition in self.state_to_transitions[state]:
                if transition.is_epsilon():
                    return False  # epsilon transition in the second component
                if transition.target in first_visited:
                    return False  # transition back from second to first component
                if transition.target not in second_visited:
                    second_queue.append(transition.target)
                if transition.accepting:
                    found_accepting = True
        visited = first_visited | second_visited
        if len(visited) < self.num_states:
            return False  # not fully connected
        return found_accepting

    def check_deterministic_transitions(self, state: int) -> bool:
        """Checks that the transitions from a state are deterministic."""
        num_assignment_transitions = Counter([
            a for transition in self.state_to_transitions[state] for a in transition.valid_assignments
        ])
        return all(c <= 1 for c in num_assignment_transitions.values())

    def complete_sink_state(self):
        if self.complete:
            return
        sink_state = self.num_states
        all_assignments = set([a.to_frozen() for a in Assignment.all_possible_assignments(tuple(self.propositions))])
        for state in range(self.num_states):
            covered_assignments = set.union(
                *[t.valid_assignments for t in self.state_to_transitions[state]]
            )
            if len(covered_assignments) != 2 ** len(self.propositions):
                # missing transitions - need to add sink state
                if not self.has_sink_state():
                    self.sink_state = sink_state
                    self.add_state(sink_state)
                    self.add_transition(sink_state, sink_state, 't', False)
                    assert self.has_sink_state()
                sink_assignments = all_assignments - covered_assignments
                sink_label = self.valid_assignments_to_label(sink_assignments)
                self.add_transition(state, sink_state, sink_label, False)
        self.complete = True

    def has_sink_state(self) -> bool:
        return self.sink_state is not None

    @staticmethod
    def valid_assignments_to_label(valid_assignments: set[FrozenAssignment]) -> str:
        assert len(valid_assignments) > 0
        formula = ' | '.join('(' + a.to_label() + ')' for a in valid_assignments)
        simplified = simplify(to_sympy(formula))
        return to_str(simplified)


@dataclass(frozen=True)
class LDBATransition:
    source: int
    target: int
    label: Optional[str]  # None for epsilon transitions
    accepting: bool
    propositions: tuple[str, ...]

    def is_epsilon(self) -> bool:
        return self.label is None

    @functools.cached_property
    def valid_assignments(self) -> set[FrozenAssignment]:
        if self.is_epsilon():
            return set()
        formula = to_sympy(self.label)
        return {a.to_frozen() for a in Assignment.all_possible_assignments(self.propositions) if a.satisfies(formula)}
