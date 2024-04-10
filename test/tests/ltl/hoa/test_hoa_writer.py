import unittest
from ltl.hoa.hoa_writer import HOAWriter
from ltl.automata.ldba import LDBA, LDBATransition


class TestHOAParser(unittest.TestCase):
    @staticmethod
    def get_hoa_text(automaton_name) -> str:
        with open(f'resources/automata/{automaton_name}.hoa', 'r') as file:
            hoa_text = file.read()
        return hoa_text

    def test_aut_1(self):
        ldba = LDBA()
        ldba.add_state(0, initial=True)
        ldba.add_state(1)
        ldba.add_state(2)
        ldba.add_transition(0, 1, 'a', False)
        ldba.add_transition(0, 2, 'b', False)
        ldba.add_transition(1, 1, 'a', True)
        ldba.add_transition(2, 2, 'b', True)
        writer = HOAWriter(ldba)
        self.assertEqual(self.get_hoa_text('writer_aut1'), writer.get_hoa())

    def test_aut_2(self):
        ldba = LDBA()
        ldba.add_state(0, initial=True)
        ldba.add_state(1)
        ldba.add_state(2)
        ldba.add_transition(0, 0, 't', False)
        ldba.add_transition(0, 1, None, False)
        ldba.add_transition(0, 2, None, False)
        ldba.add_transition(1, 1, 'a', True)
        ldba.add_transition(2, 2, 'b', True)
        writer = HOAWriter(ldba)
        self.assertEqual(self.get_hoa_text('writer_aut2'), writer.get_hoa())
