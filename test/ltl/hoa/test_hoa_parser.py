import pytest

from ltl.automata.utils import draw_ldba
from test.utils import get_resource
from test.ltl.automata.test_ldba import assert_transitions_equal
from ltl.hoa.hoa_parser import HOAParser
from ltl.automata.ldba import LDBA


def parse_automaton(automaton_name) -> LDBA:
    hoa_text = get_resource(f'{automaton_name}.hoa')
    parser = HOAParser(hoa_text)
    return parser.parse_hoa()


def test_aut_1():
    ldba = parse_automaton('aut1')
    assert 3 == ldba.num_states
    assert 0 == ldba.initial_state
    assert ldba.propositions == ('a', 'b')
    assert ldba.check_valid()
    assert 3 == len(ldba.state_to_transitions)
    assert_transitions_equal(ldba, 0, {
        (0, 0, 't', False),
        (0, 1, None, False),
        (0, 2, None, False),
    })
    assert_transitions_equal(ldba, 1, {
        (1, 1, 'b', True),
    })
    assert_transitions_equal(ldba, 2, {
        (2, 2, 'a', True),
    })


def test_aut_2():
    ldba = parse_automaton('aut2')
    assert 2 == ldba.num_states
    assert 0 == ldba.initial_state
    assert ldba.propositions == ('a', 'b', 'u')
    assert ldba.check_valid()
    assert 2 == len(ldba.state_to_transitions)
    assert_transitions_equal(ldba, 0, {
        (0, 0, 'a & b & !u', True),
        (0, 0, '!b & !u', False),
        (0, 1, '!a & b & !u', False),
    })
    assert_transitions_equal(ldba, 1, {
        (1, 1, '!a & !u', False),
        (1, 0, 'a & !u', True),
    })


def test_aut_3():
    ldba = parse_automaton('aut3')
    assert 4 == ldba.num_states
    assert 0 == ldba.initial_state
    assert ldba.propositions == ('a', 'b', 'c', 'd')
    assert ldba.check_valid()
    assert 4 == len(ldba.state_to_transitions)
    assert_transitions_equal(ldba, 0, {
        (0, 0, '!a & (b & c & !d | !b)', False),
        (0, 1, '!a & b & !c & !d', False),
        (0, 2, 'a & b & !c & !d', False),
        (0, 3, 'b & d', True),
    })
    assert_transitions_equal(ldba, 1, {
        (1, 1, '!a & !c & !d', False),
        (1, 2, 'a & !c & !d', False),
        (1, 0, '!a & c & !d', False),
        (1, 3, 'd', True),
    })
    assert_transitions_equal(ldba, 2, {
        (2, 2, '!c & !d', False),
        (2, 3, 'd', True),
    })
    assert_transitions_equal(ldba, 3, {
        (3, 3, 't', True),
    })


def test_not_alphabetical():
    ldba = parse_automaton('not_alphabetical')
    assert 3 == ldba.num_states
    assert 0 == ldba.initial_state
    assert ldba.propositions == ('a', 'b')
    assert ldba.check_valid()
    assert 3 == len(ldba.state_to_transitions)
    assert_transitions_equal(ldba, 0, {
        (0, 1, 'b & !a', False),
        (0, 2, 'a & !b', False),
    })
    assert_transitions_equal(ldba, 1, {
        (1, 1, 't', False),
    })
    assert_transitions_equal(ldba, 2, {
        (2, 2, 't', True),
    })


############################
# Error handling tests
############################
def test_missing_header_fields():
    with pytest.raises(ValueError) as context:
        parse_automaton('missing_header_fields')
    assert str(context.value) == 'Error parsing HOA. Missing required header field `AP`.'


def test_missing_body():
    with pytest.raises(ValueError) as context:
        parse_automaton('missing_body')
    assert 'Error parsing HOA. Reached end of input. Expecting "--BODY--".' == str(context.value)


def test_missing_state_line():
    with pytest.raises(ValueError) as context:
        parse_automaton('missing_state')
    assert 'Error parsing HOA at line 9. Expected a state line.' == str(context.value)
