import unittest
from ltl.hoa.hoa_parser import HOAParser
from ltl.automata.ldba import LDBA, LDBATransition


class TestHOAParser(unittest.TestCase):

    @staticmethod
    def parse_automaton(automaton_name) -> LDBA:
        with open(f'resources/automata/{automaton_name}.hoa', 'r') as file:
            hoa_text = file.read()
        parser = HOAParser(hoa_text)
        return parser.parse_hoa()

    def assert_transitions_equal(self, ldba: LDBA, state: int, expected: set[LDBATransition]):
        self.assertEqual(expected, set(ldba.state_to_transitions[state]))

    def test_aut_1(self):
        ldba = self.parse_automaton('aut1')
        self.assertEqual(3, ldba.num_states)
        self.assertEqual(0, ldba.initial_state)
        self.assertEqual({'a', 'b'}, ldba.propositions)
        self.assertTrue(ldba.check_valid())
        self.assertEqual(3, len(ldba.state_to_transitions))
        self.assert_transitions_equal(ldba, 0, {
            LDBATransition(0, 0, 't', False),
            LDBATransition(0, 1, None, False),
            LDBATransition(0, 2, None, False),
        })
        self.assert_transitions_equal(ldba, 1, {
            LDBATransition(1, 1, 'b', True),
        })
        self.assert_transitions_equal(ldba, 2, {
            LDBATransition(2, 2, 'a', True),
        })

    def test_aut_2(self):
        ldba = self.parse_automaton('aut2')
        self.assertEqual(2, ldba.num_states)
        self.assertEqual(0, ldba.initial_state)
        self.assertEqual({'a', 'b', 'u'}, ldba.propositions)
        self.assertTrue(ldba.check_valid())
        self.assertEqual(2, len(ldba.state_to_transitions))
        self.assert_transitions_equal(ldba, 0, {
            LDBATransition(0, 0, 'a & b & !u', True),
            LDBATransition(0, 0, '!b & !u', False),
            LDBATransition(0, 1, '!a & b & !u', False),
        })
        self.assert_transitions_equal(ldba, 1, {
            LDBATransition(1, 1, '!a & !u', False),
            LDBATransition(1, 0, 'a & !u', True),
        })

    def test_aut_3(self):
        ldba = self.parse_automaton('aut3')
        self.assertEqual(4, ldba.num_states)
        self.assertEqual(0, ldba.initial_state)
        self.assertEqual({'a', 'b', 'c', 'd'}, ldba.propositions)
        self.assertTrue(ldba.check_valid())
        self.assertEqual(4, len(ldba.state_to_transitions))
        self.assert_transitions_equal(ldba, 0, {
            LDBATransition(0, 0, '!a & (b & c & !d | !b)', False),
            LDBATransition(0, 1, '!a & b & !c & !d', False),
            LDBATransition(0, 2, 'a & b & !c & !d', False),
            LDBATransition(0, 3, 'b & d', True),
        })
        self.assert_transitions_equal(ldba, 1, {
            LDBATransition(1, 1, '!a & !c & !d', False),
            LDBATransition(1, 2, 'a & !c & !d', False),
            LDBATransition(1, 0, '!a & c & !d', False),
            LDBATransition(1, 3, 'd', True),
        })
        self.assert_transitions_equal(ldba, 2, {
            LDBATransition(2, 2, '!c & !d', False),
            LDBATransition(2, 3, 'd', True),
        })
        self.assert_transitions_equal(ldba, 3, {
            LDBATransition(3, 3, 't', True),
        })

    ############################
    # Error handling tests
    ############################
    def test_missing_header_fields(self):
        with self.assertRaises(ValueError) as context:
            self.parse_automaton('missing_header_fields')
        self.assertTrue(
            str(context.exception).startswith('Error parsing HOA at line 5. Missing required header fields:'))

    def test_missing_body(self):
        with self.assertRaises(ValueError) as context:
            self.parse_automaton('missing_body')
        self.assertEqual('Error parsing HOA. Reached end of input. Expecting "--BODY--".', str(context.exception))

    def test_missing_state_line(self):
        with self.assertRaises(ValueError) as context:
            self.parse_automaton('missing_state')
        self.assertEqual('Error parsing HOA at line 9. Expected a state line.', str(context.exception))


if __name__ == '__main__':
    unittest.main()
