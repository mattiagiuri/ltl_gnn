import unittest
from ltl.automata.ldba import LDBA


class TestLDBA(unittest.TestCase):
    def setUp(self):
        self.ldba = LDBA()

    def test_set_states_with_valid_input(self):
        self.ldba.set_states(5, 0)
        self.assertEqual(self.ldba.num_states, 5)
        self.assertEqual(self.ldba.initial_state, 0)

    def test_set_states_with_negative_num_states(self):
        with self.assertRaises(ValueError):
            self.ldba.set_states(-5, 0)

    def test_set_states_with_invalid_initial_state(self):
        with self.assertRaises(ValueError):
            self.ldba.set_states(5, 6)

    def test_add_transition_with_valid_input(self):
        self.ldba.set_states(2, 0)
        self.ldba.add_transition(0, 1, 'a', True)
        self.assertEqual(len(self.ldba.state_to_transitions[0]), 1)

    def test_check_valid_with_valid_ldba(self):
        self.ldba.set_states(2, 0)
        self.ldba.add_transition(0, 1, 'a', False)
        self.ldba.add_transition(1, 1, 't', True)
        self.assertTrue(self.ldba.check_valid())

    def test_check_valid_with_empty_first(self):
        self.ldba.set_states(1, 0)
        self.ldba.add_transition(0, 0, 'a', True)
        self.assertTrue(self.ldba.check_valid())

    def test_check_valid_without_accepting(self):
        self.ldba.set_states(2, 0)
        self.ldba.add_transition(0, 1, None, False)
        self.assertFalse(self.ldba.check_valid())

    def test_check_valid_with_epsilon_in_first(self):
        self.ldba.set_states(4, 0)
        self.ldba.add_transition(0, 2, 'a', False)
        self.ldba.add_transition(2, 1, 'b', False)
        self.ldba.add_transition(2, 3, None, False)
        self.ldba.add_transition(3, 3, 'a', True)
        self.assertTrue(self.ldba.check_valid())
        self.ldba.add_transition(0, 1, None, False)
        self.assertFalse(self.ldba.check_valid())

    def test_check_valid_with_transition_from_second_to_first(self):
        self.ldba.set_states(2, 0)
        self.ldba.add_transition(0, 0, 'a', False)
        self.ldba.add_transition(0, 1, None, False)
        self.ldba.add_transition(1, 1, 'b', True)
        self.assertTrue(self.ldba.check_valid())
        self.ldba.add_transition(1, 0, 'a', False)
        self.assertFalse(self.ldba.check_valid())

    def test_check_valid_with_epsilon_in_second(self):
        self.ldba.set_states(3, 0)
        self.ldba.add_transition(0, 1, None, False)
        self.ldba.add_transition(1, 1, 'a', True)
        self.assertTrue(self.ldba.check_valid())
        self.ldba.add_transition(1, 2, None, False)
        self.assertFalse(self.ldba.check_valid())

    def test_check_valid_with_accepting_in_first(self):
        self.ldba.set_states(2, 0)
        self.ldba.add_transition(0, 0, 'a', True)
        self.ldba.add_transition(0, 1, None, False)
        self.ldba.add_transition(1, 1, 'b', True)
        self.assertFalse(self.ldba.check_valid())

    def test_check_deterministic_transitions_with_deterministic_transitions(self):
        self.ldba.set_states(2, 0)
        self.ldba.add_transition(0, 1, 'a', False)
        self.ldba.add_transition(0, 1, 'b', False)
        self.assertTrue(self.ldba.check_deterministic_transitions(0))

    def test_check_deterministic_transitions_with_non_deterministic_transitions(self):
        self.ldba.set_states(2, 0)
        self.ldba.add_transition(0, 1, 'a & b', False)
        self.ldba.add_transition(0, 1, 'b & a', False)
        self.assertFalse(self.ldba.check_deterministic_transitions(0))

    def test_propositions(self):
        self.ldba.set_states(3, 0)
        self.ldba.add_transition(0, 0, 'a & b', False)
        self.ldba.add_transition(0, 1, '!c', False)
        self.ldba.add_transition(1, 1, 'var | e', False)
        self.ldba.add_transition(1, 2, 'c', False)
        self.ldba.add_transition(2, 1, 'f & ((g_2 => !h) | !asd)', False)
        self.assertEqual({'a', 'b', 'c', 'var', 'e', 'f', 'g_2', 'h', 'asd'},
                         self.ldba.propositions)


if __name__ == '__main__':
    unittest.main()
