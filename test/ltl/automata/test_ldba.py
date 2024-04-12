from ltl.automata.ldba import LDBA, LDBATransition


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
    add_states(ldba, 2, 0)
    ldba.add_transition(0, 1, 'a & b', False)
    ldba.add_transition(0, 1, 'b & a', False)
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
    ldba.add_transition(1, 1, 'a | b | !a', True)
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
        (1, 1, 'a | b | !a', True),
    })


def assert_transitions_equal(ldba: LDBA, state: int, expected: set[tuple[int, int, str, bool]]):
    expected = {LDBATransition(*e, propositions=ldba.propositions) for e in expected}
    assert set(ldba.state_to_transitions[state]) == expected
