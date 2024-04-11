import pytest
from ltl.hoa.hoa_parser import HOAParser
from ltl.automata.ldba import LDBA, LDBATransition


def parse_automaton(automaton_name) -> LDBA:
    with open(f'resources/automata/{automaton_name}.hoa', 'r') as file:
        hoa_text = file.read()
    parser = HOAParser(hoa_text)
    return parser.parse_hoa()


def assert_transitions_equal(ldba: LDBA, state: int, expected: set[LDBATransition]):
    assert expected == set(ldba.state_to_transitions[state])


def test_aut_1():
    ldba = parse_automaton('aut1')
    assert 3 == ldba.num_states
    assert 0 == ldba.initial_state
    assert {'a', 'b'} == ldba.propositions
    assert ldba.check_valid()
    assert 3 == len(ldba.state_to_transitions)
    assert_transitions_equal(ldba, 0, {
        LDBATransition(0, 0, 't', False),
        LDBATransition(0, 1, None, False),
        LDBATransition(0, 2, None, False),
    })
    assert_transitions_equal(ldba, 1, {
        LDBATransition(1, 1, 'b', True),
    })
    assert_transitions_equal(ldba, 2, {
        LDBATransition(2, 2, 'a', True),
    })


def test_aut_2():
    ldba = parse_automaton('aut2')
    assert 2 == ldba.num_states
    assert 0 == ldba.initial_state
    assert {'a', 'b', 'u'} == ldba.propositions
    assert ldba.check_valid()
    assert 2 == len(ldba.state_to_transitions)
    assert_transitions_equal(ldba, 0, {
        LDBATransition(0, 0, 'a & b & !u', True),
        LDBATransition(0, 0, '!b & !u', False),
        LDBATransition(0, 1, '!a & b & !u', False),
    })
    assert_transitions_equal(ldba, 1, {
        LDBATransition(1, 1, '!a & !u', False),
        LDBATransition(1, 0, 'a & !u', True),
    })


def test_aut_3():
    ldba = parse_automaton('aut3')
    assert 4 == ldba.num_states
    assert 0 == ldba.initial_state
    assert {'a', 'b', 'c', 'd'} == ldba.propositions
    assert ldba.check_valid()
    assert 4 == len(ldba.state_to_transitions)
    assert_transitions_equal(ldba, 0, {
        LDBATransition(0, 0, '!a & (b & c & !d | !b)', False),
        LDBATransition(0, 1, '!a & b & !c & !d', False),
        LDBATransition(0, 2, 'a & b & !c & !d', False),
        LDBATransition(0, 3, 'b & d', True),
    })
    assert_transitions_equal(ldba, 1, {
        LDBATransition(1, 1, '!a & !c & !d', False),
        LDBATransition(1, 2, 'a & !c & !d', False),
        LDBATransition(1, 0, '!a & c & !d', False),
        LDBATransition(1, 3, 'd', True),
    })
    assert_transitions_equal(ldba, 2, {
        LDBATransition(2, 2, '!c & !d', False),
        LDBATransition(2, 3, 'd', True),
    })
    assert_transitions_equal(ldba, 3, {
        LDBATransition(3, 3, 't', True),
    })


############################
# Error handling tests
############################
def test_missing_header_fields():
    with pytest.raises(ValueError) as context:
        parse_automaton('missing_header_fields')
    assert str(context.value).startswith('Error parsing HOA at line 5. Missing required header fields:')


def test_missing_body():
    with pytest.raises(ValueError) as context:
        parse_automaton('missing_body')
    assert 'Error parsing HOA. Reached end of input. Expecting "--BODY--".' == str(context.value)


def test_missing_state_line():
    with pytest.raises(ValueError) as context:
        parse_automaton('missing_state')
    assert 'Error parsing HOA at line 9. Expected a state line.' == str(context.value)
