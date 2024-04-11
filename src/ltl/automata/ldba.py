from collections import Counter
from dataclasses import dataclass
from typing import Optional

from sympy import simplify_logic

from utils import to_sympy


class LDBA:
    def __init__(self):
        self.num_states = 0
        self.num_transitions = 0
        self.initial_state = None
        self.state_to_transitions: dict[int, list[LDBATransition]] = {}
        self.state_to_incoming_transitions: dict[int, list[LDBATransition]] = {}
        self.propositions = set()

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

    def add_transition(self, source: int, target: int, label: Optional[str], accepting: bool):
        if source < 0 or source >= self.num_states:
            raise ValueError('Source state must be a valid state index.')
        if target < 0 or target >= self.num_states:
            raise ValueError('Target state must be a valid state index.')
        transition = LDBATransition(source, target, label, accepting)
        self.num_transitions += 1
        self.state_to_transitions[source].append(transition)
        self.state_to_incoming_transitions[target].append(transition)
        self.update_propositions(label)

    def update_propositions(self, label: Optional[str]):
        if label is None:
            return
        props = [str(a) for a in to_sympy(label).atoms() if str(a) != 't']
        self.propositions.update(props)

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
                if transition.label is None:
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
                if transition.label is None:
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
        num_label_transitions = Counter([
            simplify_logic(to_sympy(transition.label)) for transition in self.state_to_transitions[state]
            if transition.label is not None
        ])
        if any(c > 1 for c in num_label_transitions.values()):
            return False
        return True

    def complete_sink_state(self):
        sink_state = self.num_states
        added_sink_state = False
        # TODO


@dataclass(frozen=True)
class LDBATransition:
    source: int
    target: int
    label: Optional[str]  # None for epsilon transitions
    accepting: bool
