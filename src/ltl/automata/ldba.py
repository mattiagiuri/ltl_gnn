from collections import Counter
from dataclasses import dataclass
from typing import Optional

import spot


class LDBA:
    def __init__(self):
        self.num_states = 0
        self.initial_state = None
        self.state_to_transitions: dict[int, list[LDBATransition]] = {}
        self.propositions = set()

    def set_states(self, num_states: int, initial: int):
        if num_states <= 0:
            raise ValueError('Number of states must be positive.')
        if initial < 0 or initial >= num_states:
            raise ValueError('Initial state must be a valid state index.')
        self.num_states = num_states
        self.initial_state = initial
        for state in range(num_states):
            self.state_to_transitions[state] = []

    def add_transition(self, source: int, target: int, label: Optional[str], accepting: bool):
        if source < 0 or source >= self.num_states:
            raise ValueError('Source state must be a valid state index.')
        if target < 0 or target >= self.num_states:
            raise ValueError('Target state must be a valid state index.')
        transition = LDBATransition(source, target, label, accepting)
        self.state_to_transitions[source].append(transition)
        self.update_propositions(label)

    def update_propositions(self, label: Optional[str]):
        if label is None:
            return
        props = [p.ap_name() for p in spot.atomic_prop_collect(spot.formula(label))]
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
        return found_accepting

    def check_deterministic_transitions(self, state: int) -> bool:
        """Checks that the transitions from a state are deterministic."""
        num_label_transitions = Counter([
            # Spot implements equivalence checking for boolean formulae
            spot.formula(transition.label) for transition in self.state_to_transitions[state]
            if transition.label is not None
        ])
        if any(c > 1 for c in num_label_transitions.values()):
            return False
        return True


@dataclass
class LDBATransition:
    source: int
    target: int
    label: Optional[str]  # None for epsilon transitions
    accepting: bool
