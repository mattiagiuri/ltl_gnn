import pytest

from ltl.automata import LDBA, LDBATransition
from ltl.automata.ldba import SCC
from ltl.logic import Assignment


def add_states(ldba, num_states: int, initial_state: int):
    for state in range(num_states):
        ldba.add_state(state, initial=state == initial_state)


def test_set_states_with_valid_input():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 5, 0)
    assert ldba.num_states == 5
    assert ldba.initial_state == 0


def test_add_transition_with_valid_input():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 2, 0)
    ldba.add_transition(0, 1, 'a', True)
    assert ldba.num_transitions == 1
    assert len(ldba.state_to_transitions[0]) == 1
    assert ldba.state_to_transitions[0][0].label == 'a'
    assert len(ldba.state_to_incoming_transitions[1]) == 1
    assert ldba.state_to_incoming_transitions[1][0].label == 'a'


def test_cannot_add_two_transitions_with_same_target():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 2, 0)
    ldba.add_transition(0, 1, 'a', False)
    with pytest.raises(ValueError):
        ldba.add_transition(0, 1, 'b', False)
    ldba.add_transition(0, 1, 'b', True)  # should work


def test_check_valid_with_valid_ldba():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 2, 0)
    ldba.add_transition(0, 1, 'a', False)
    ldba.add_transition(1, 1, 't', True)
    assert ldba.check_valid()

    ldba = LDBA({'a', 'b', 'c', 'd'})
    add_states(ldba, 3, 0)
    ldba.add_transition(0, 1, 'a & !b', False)
    ldba.add_transition(0, 2, 'b & !a', False)
    ldba.add_transition(1, 2, 'c', False)
    ldba.add_transition(2, 1, 'd', True)
    assert ldba.check_valid()


def test_check_valid_with_empty_first():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 1, 0)
    ldba.add_transition(0, 0, 'a', True)
    assert ldba.check_valid()


def test_check_valid_without_accepting():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 2, 0)
    ldba.add_transition(0, 1, None, False)
    assert not ldba.check_valid()


def test_check_valid_with_epsilon_in_first():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 4, 0)
    ldba.add_transition(0, 2, 'a', False)
    ldba.add_transition(2, 1, 'b', False)
    ldba.add_transition(2, 3, None, False)
    ldba.add_transition(3, 3, 'a', True)
    assert ldba.check_valid()
    ldba.add_transition(0, 1, None, False)
    assert not ldba.check_valid()


def test_check_valid_with_transition_from_second_to_first():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 2, 0)
    ldba.add_transition(0, 0, 'a', False)
    ldba.add_transition(0, 1, None, False)
    ldba.add_transition(1, 1, 'b', True)
    assert ldba.check_valid()
    ldba.add_transition(1, 0, 'a', False)
    assert not ldba.check_valid()


def test_check_valid_with_epsilon_in_second():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 3, 0)
    ldba.add_transition(0, 1, None, False)
    ldba.add_transition(1, 1, 'a', True)
    ldba.add_transition(1, 2, None, False)
    assert not ldba.check_valid()


def test_check_valid_with_accepting_in_first():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 2, 0)
    ldba.add_transition(0, 0, 'a', True)
    ldba.add_transition(0, 1, None, False)
    ldba.add_transition(1, 1, 'b', True)
    assert not ldba.check_valid()


def test_check_valid_not_connected():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 4, 0)
    ldba.add_transition(0, 1, 'a', True)
    ldba.add_transition(2, 3, 'b', False)
    assert not ldba.check_valid()


def test_check_deterministic_transitions_with_deterministic_transitions():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 3, 0)
    ldba.add_transition(0, 1, 'a & !b', False)
    ldba.add_transition(0, 2, 'b & !a', False)
    assert ldba.check_deterministic_transitions(0)


def test_check_deterministic_transitions_with_non_deterministic_transitions():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 3, 0)
    ldba.add_transition(0, 1, 'a & b', False)
    ldba.add_transition(0, 2, 'b & a', False)
    assert not ldba.check_deterministic_transitions(0)
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 3, 0)
    ldba.add_transition(0, 1, '!(a | b)', False)
    ldba.add_transition(0, 2, '!a & !b', False)
    assert not ldba.check_deterministic_transitions(0)
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 3, 0)
    ldba.add_transition(0, 1, 'a', False)
    ldba.add_transition(0, 2, 'b', False)
    assert not ldba.check_deterministic_transitions(0)


def test_complete_sink_state1():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 2, 0)
    ldba.add_transition(0, 0, 't', False)
    ldba.add_transition(0, 1, None, False)
    ldba.add_transition(1, 1, 'a', True)
    assert ldba.check_valid()
    ldba.complete_sink_state()

    assert ldba.check_valid()
    assert ldba.has_sink_state()
    assert ldba.sink_state == 2
    assert len(ldba.state_to_transitions) == 3
    assert_transitions_equal(ldba, 0, {
        (0, 0, 't', False),
        (0, 1, None, False),
    })
    assert_transitions_equal(ldba, 1, {
        (1, 1, 'a', True),
        (1, 2, '!a', False)
    })
    assert_transitions_equal(ldba, 2, {
        (2, 2, 't', False)
    })


def test_complete_sink_state2():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 2, 0)
    ldba.add_transition(0, 0, 'a & !b', False)
    ldba.add_transition(0, 1, 'b & !a', False)
    ldba.add_transition(1, 1, 'a', True)
    assert ldba.check_valid()
    ldba.complete_sink_state()

    assert ldba.check_valid()
    assert ldba.has_sink_state()
    assert ldba.sink_state == 2
    assert len(ldba.state_to_transitions) == 3
    assert_transitions_equal(ldba, 0, {
        (0, 0, 'a & !b', False),
        (0, 1, 'b & !a', False),
        (0, 2, '(a & b) | (!a & !b)', False)
    })
    assert_transitions_equal(ldba, 1, {
        (1, 1, 'a', True),
        (1, 2, '!a', False)
    })
    assert_transitions_equal(ldba, 2, {
        (2, 2, 't', False)
    })


def test_complete_sink_state3():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 3, 0)
    ldba.add_transition(0, 0, 't', False)
    ldba.add_transition(0, 1, None, False)
    ldba.add_transition(0, 2, None, False)
    ldba.add_transition(1, 1, 'b', True)
    ldba.add_transition(2, 2, 'a', True)
    assert ldba.check_valid()
    ldba.complete_sink_state()

    assert ldba.check_valid()
    assert ldba.has_sink_state()
    assert ldba.sink_state == 3
    assert len(ldba.state_to_transitions) == 4
    assert_transitions_equal(ldba, 0, {
        (0, 0, 't', False),
        (0, 1, None, False),
        (0, 2, None, False)
    })
    assert_transitions_equal(ldba, 1, {
        (1, 1, 'b', True),
        (1, 3, '!b', False)
    })
    assert_transitions_equal(ldba, 2, {
        (2, 2, 'a', True),
        (2, 3, '!a', False)
    })
    assert_transitions_equal(ldba, 3, {
        (3, 3, 't', False)
    })


def test_complete_sink_state_no_change():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 2, 0)
    ldba.add_transition(0, 0, 'a', False)
    ldba.add_transition(0, 1, '!a', False)
    ldba.add_transition(1, 1, 't', True)
    assert ldba.check_valid()
    ldba.complete_sink_state()
    assert ldba.check_valid()
    assert not ldba.has_sink_state()
    assert ldba.sink_state is None
    assert len(ldba.state_to_transitions) == 2
    assert_transitions_equal(ldba, 0, {
        (0, 0, 'a', False),
        (0, 1, '!a', False)})
    assert_transitions_equal(ldba, 1, {
        (1, 1, 't', True),
    })


def test_prune_impossible_transitions():
    ldba = LDBA({'a', 'b', 'c'}, )
    add_states(ldba, 4, 0)
    ldba.add_transition(0, 1, 'a & !b', False)
    ldba.add_transition(0, 2, 'b & !a', False)
    ldba.add_transition(1, 2, 'a & b & c', False)
    ldba.add_transition(1, 1, 'a & b & !c', True)
    ldba.add_transition(1, 3, '!a & c', True)
    ldba.add_transition(2, 1, 'c', True)
    ldba.add_transition(3, 3, 't', True)
    assert ldba.check_valid()
    ldba.prune_impossible_transitions(impossible_assignments={
        Assignment({'a': True, 'b': True, 'c': False}).to_frozen(),
        Assignment({'a': True, 'b': True, 'c': True}).to_frozen(),
    })
    assert ldba.num_transitions == 5
    assert_transitions_equal(ldba, 0, {
        (0, 1, 'a & !b', False),
        (0, 2, 'b & !a', False),
    })
    assert_transitions_equal(ldba, 1, {
        (1, 3, '!a & c', True),
    })
    assert_transitions_equal(ldba, 2, {
        (2, 1, 'c', True),
    })
    assert_transitions_equal(ldba, 3, {
        (3, 3, 't', True),
    })


def test_positive_label():
    props = ('a', 'b')
    t = LDBATransition(0, 1, 'a & !b', True, props)
    assert t.positive_label == 'a'
    t = LDBATransition(0, 1, '!a & b', True, props)
    assert t.positive_label == 'b'
    t = LDBATransition(0, 1, 'a & b', True, props)
    assert t.positive_label == 'a&b' or t.positive_label == 'b&a'
    t = LDBATransition(0, 1, None, False, props)
    assert t.positive_label == 'ε'
    t = LDBATransition(0, 1, '!a & !b', False, props)
    assert t.positive_label == '{}'


def assert_transitions_equal(ldba: LDBA, state: int, expected: set[tuple[int, int, str, bool]]):
    expected = {LDBATransition(*e, propositions=ldba.propositions) for e in expected}
    assert set(ldba.state_to_transitions[state]) == expected


def test_get_next_state():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 4, 0)
    ldba.add_transition(0, 1, 'a', False)
    ldba.add_transition(0, 0, '!a', False)
    ldba.add_transition(1, 2, 'a & b', True)
    ldba.add_transition(1, 3, 'a & !b', False)
    ldba.add_transition(2, 3, 'b', True)
    ldba.add_transition(3, 3, 't', True)
    ldba.complete_sink_state()
    assert ldba.check_valid()
    assert ldba.sink_state == 4

    assert_next_states(ldba, 0, [0, 1, 0, 1], [False, False, False, False])
    assert_next_states(ldba, 1, [4, 3, 4, 2], [False, False, False, True])
    assert_next_states(ldba, 2, [4, 4, 3, 3], [False, False, True, True])
    assert_next_states(ldba, 3, [3, 3, 3, 3], [True, True, True, True])
    assert_next_states(ldba, 4, [4, 4, 4, 4], [False, False, False, False])


def assert_next_states(ldba: LDBA, state: int, expected_states: list[int], expected_accepting: list[bool]):
    """Asserts that the next states are as expected for the given state. The expected states should be given in order of
    the assignments {}, {a}, {b}, {a,b}."""
    for i, assignment in enumerate([set(), {'a'}, {'b'}, {'a', 'b'}]):
        assert ldba.get_next_state(state, assignment) == (expected_states[i], expected_accepting[i])


def test_find_sccs1():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 12, 0)
    transitions = [
        (0, 1, 'a', False),
        (1, 2, 't', True),
        (2, 3, 't', False),
        (3, 2, 't', False),
        (0, 4, '!a', False),
        (4, 5, 't', True),
        (5, 6, 'a', False),
        (5, 3, '!a', False),
        (6, 7, 'a', False),
        (6, 8, '!a', False),
        (7, 5, 'a', False),
        (7, 4, '!a', False),
        (8, 9, 't', True),
        (9, 10, 't', False),
        (10, 11, 't', False),
        (11, 8, 't', False),
    ]
    for t in transitions:
        ldba.add_transition(*t)
    ldba.complete_sink_state()
    assert ldba.check_valid()
    ldba.compute_sccs()
    expected = {
        SCC(states=frozenset({2, 3}), accepting=False, bottom=True),
        SCC(states=frozenset({4, 5, 6, 7}), accepting=True, bottom=False),
        SCC(states=frozenset({8, 9, 10, 11}), accepting=True, bottom=True),
        SCC(states=frozenset({1}), accepting=False, bottom=False),
        SCC(states=frozenset({0}), accepting=False, bottom=False),
    }
    assert (expected == set(ldba.state_to_scc.values()))


def test_find_sccs2():
    ldba = LDBA({'a', 'b'})
    add_states(ldba, 4, 0)
    transitions = [
        (0, 1, 'a & !b', False),
        (0, 0, 'b & !a', True),
        (1, 2, 'b', False),
        (2, 3, 'c', False),
        (3, 1, 'd', True),
    ]
    for t in transitions:
        ldba.add_transition(*t)
    ldba.complete_sink_state()
    assert ldba.check_valid()
    ldba.compute_sccs()
    expected = {
        SCC(states=frozenset({4}), accepting=False, bottom=True),
        SCC(states=frozenset({0}), accepting=True, bottom=False),
        SCC(states=frozenset({1, 2, 3}), accepting=True, bottom=False)
    }
    assert (expected == set(ldba.state_to_scc.values()))
