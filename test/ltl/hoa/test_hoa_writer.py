from ltl.hoa import HOAWriter
from ltl.automata import LDBA
from test.utils import get_resource


def get_hoa_text(automaton_name) -> str:
    return get_resource(f'{automaton_name}.hoa')


def test_aut_1():
    ldba = LDBA({'a', 'b'})
    ldba.add_state(0, initial=True)
    ldba.add_state(1)
    ldba.add_state(2)
    ldba.add_transition(0, 1, 'a', False)
    ldba.add_transition(0, 2, 'b', False)
    ldba.add_transition(1, 1, 'a', True)
    ldba.add_transition(2, 2, 'b', True)
    writer = HOAWriter(ldba)
    assert get_hoa_text('writer_aut1') == writer.get_hoa()


def test_aut_2():
    ldba = LDBA({'a', 'b'})
    ldba.add_state(0, initial=True)
    ldba.add_state(1)
    ldba.add_state(2)
    ldba.add_transition(0, 0, 't', False)
    ldba.add_transition(0, 1, None, False)
    ldba.add_transition(0, 2, None, False)
    ldba.add_transition(1, 1, 'a', True)
    ldba.add_transition(2, 2, 'b', True)
    writer = HOAWriter(ldba)
    assert get_hoa_text('writer_aut2') == writer.get_hoa()
