from ltl.automata.ldba import LDBA


def add_states(ldba, num_states: int, initial_state: int):
    for state in range(num_states):
        ldba.add_state(state, initial=state == initial_state)


def test_set_states_with_valid_input():
    ldba = LDBA()
    add_states(ldba, 5, 0)
    assert ldba.num_states == 5
    assert ldba.initial_state == 0


def test_add_transition_with_valid_input():
    ldba = LDBA()
    add_states(ldba, 2, 0)
    ldba.add_transition(0, 1, 'a', True)
    assert len(ldba.state_to_transitions[0]) == 1


def test_check_valid_with_valid_ldba():
    ldba = LDBA()
    add_states(ldba, 2, 0)
    ldba.add_transition(0, 1, 'a', False)
    ldba.add_transition(1, 1, 't', True)
    assert ldba.check_valid()

    ldba = LDBA()
    add_states(ldba, 3, 0)
    ldba.add_transition(0, 1, 'a', False)
    ldba.add_transition(0, 2, 'b', False)
    ldba.add_transition(1, 2, 'c', False)
    ldba.add_transition(2, 1, 'd', True)
    assert ldba.check_valid()


def test_check_valid_with_empty_first():
    ldba = LDBA()
    add_states(ldba, 1, 0)
    ldba.add_transition(0, 0, 'a', True)
    assert ldba.check_valid()


def test_check_valid_without_accepting():
    ldba = LDBA()
    add_states(ldba, 2, 0)
    ldba.add_transition(0, 1, None, False)
    assert not ldba.check_valid()


def test_check_valid_with_epsilon_in_first():
    ldba = LDBA()
    add_states(ldba, 4, 0)
    ldba.add_transition(0, 2, 'a', False)
    ldba.add_transition(2, 1, 'b', False)
    ldba.add_transition(2, 3, None, False)
    ldba.add_transition(3, 3, 'a', True)
    assert ldba.check_valid()
    ldba.add_transition(0, 1, None, False)
    assert not ldba.check_valid()


def test_check_valid_with_transition_from_second_to_first():
    ldba = LDBA()
    add_states(ldba, 2, 0)
    ldba.add_transition(0, 0, 'a', False)
    ldba.add_transition(0, 1, None, False)
    ldba.add_transition(1, 1, 'b', True)
    assert ldba.check_valid()
    ldba.add_transition(1, 0, 'a', False)
    assert not ldba.check_valid()


def test_check_valid_with_epsilon_in_second():
    ldba = LDBA()
    add_states(ldba, 3, 0)
    ldba.add_transition(0, 1, None, False)
    ldba.add_transition(1, 1, 'a', True)
    assert ldba.check_valid()
    ldba.add_transition(1, 2, None, False)
    assert not ldba.check_valid()


def test_check_valid_with_accepting_in_first():
    ldba = LDBA()
    add_states(ldba, 2, 0)
    ldba.add_transition(0, 0, 'a', True)
    ldba.add_transition(0, 1, None, False)
    ldba.add_transition(1, 1, 'b', True)
    assert not ldba.check_valid()


def test_check_deterministic_transitions_with_deterministic_transitions():
    ldba = LDBA()
    add_states(ldba, 2, 0)
    ldba.add_transition(0, 1, 'a', False)
    ldba.add_transition(0, 1, 'b', False)
    assert ldba.check_deterministic_transitions(0)


def test_check_deterministic_transitions_with_non_deterministic_transitions():
    ldba = LDBA()
    add_states(ldba, 2, 0)
    ldba.add_transition(0, 1, 'a & b', False)
    ldba.add_transition(0, 1, 'b & a', False)
    assert not ldba.check_deterministic_transitions(0)


def test_propositions():
    ldba = LDBA()
    add_states(ldba, 3, 0)
    ldba.add_transition(0, 0, 'a & b', False)
    ldba.add_transition(0, 1, '!c', False)
    ldba.add_transition(1, 1, 'var | e', False)
    ldba.add_transition(1, 2, 'c', False)
    ldba.add_transition(2, 1, 'f & ((g_2 => !h) | !asd)', False)
    assert ldba.propositions == {'a', 'b', 'c', 'var', 'e', 'f', 'g_2', 'h', 'asd'}
