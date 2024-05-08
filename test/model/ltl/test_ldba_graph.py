import pytest
import torch

from ltl.automata import LDBA, LDBATransition
from ltl.logic import Assignment
from model.ltl.ldba_graph import LDBAGraph
from visualize.visualize import draw_ldba_graph, draw_ldba


def assert_edge_index_equal(edge_index1, edge_index2):
    assert edge_index1.shape == edge_index2.shape
    edges1 = set(map(tuple, edge_index1.t().tolist()))
    for edge2 in edge_index2.t().tolist():
        assert tuple(edge2) in edges1


def test_from_ldba1():
    ldba = LDBA({'a', 'b'}, formula='')
    ldba.add_state(0, True)
    ldba.add_state(1, False)
    ldba.add_state(2, False)
    ldba.add_transition(0, 1, 'a', False)
    ldba.add_transition(1, 2, 'b', True)
    ldba.add_transition(2, 2, 't', True)
    ldba.complete_sink_state()
    assert ldba.check_valid()
    ldba.compute_sccs()
    _, g = LDBAGraph.from_ldba(ldba, 0)
    # draw_ldba_graph(g, features=False)


def test_from_ldba2():
    ldba = LDBA({'a'}, formula='')
    ldba.add_state(0, True)
    ldba.add_state(1, False)
    ldba.add_state(2, False)
    ldba.add_transition(0, 1, 'a', False)
    ldba.add_transition(0, 2, '!a', False)
    ldba.add_transition(2, 1, 'a', False)
    ldba.add_transition(1, 1, 'a', True)
    ldba.complete_sink_state()
    assert ldba.check_valid()
    ldba.compute_sccs()
    _, g = LDBAGraph.from_ldba(ldba, 0)
    # draw_ldba_graph(g, features=False)


def test_from_ldba3():
    ldba = LDBA({'a', 'b', 'c', 'd'}, formula='')
    ldba.add_state(0, True)
    ldba.add_state(1, False)
    ldba.add_state(2, False)
    ldba.add_state(3, False)
    ldba.add_transition(0, 0, 'a', True)
    ldba.add_transition(0, 1, '!a', False)
    ldba.add_transition(1, 0, '!b', False)
    ldba.add_transition(1, 2, 'b', False)
    ldba.add_transition(2, 3, 'c', False)
    ldba.add_transition(3, 1, 'd', True)
    ldba.complete_sink_state()
    ldba.prune_impossible_transitions(Assignment.more_than_one_true_proposition(set(ldba.propositions)))
    assert ldba.check_valid()
    ldba.compute_sccs()
    pos, neg = LDBAGraph.from_ldba(ldba, 0)

    # TODO: GFa & GFb & G (!a | F g)


def test_get_features():
    propositions = ('a', 'b')
    assignments = Assignment.all_possible_assignments(propositions)
    # components: {!a, !b}, {!a, b}, {a, !b}, {a, b}, eps, acc
    features = LDBAGraph.get_features(LDBATransition(0, 1, 'a & b', False, propositions), assignments)
    assert features == [0., 0., 0., 1., 0., 0.]
    features = LDBAGraph.get_features(LDBATransition(0, 1, 'a', True, propositions), assignments)
    assert features== [0., 0., 1., 1., 0., 1.]
    features = LDBAGraph.get_features(LDBATransition(0, 1, None, False, propositions), assignments)
    assert features == [0., 0., 0., 0., 1., 0.]

    propositions = ('a', 'b', 'c')
    assignments = Assignment.all_possible_assignments(propositions)
    # components:
    # {!a, !b, !c}, {!a, !b, c}, {!a, b, !c}, {!a, b, c}, {a, !b, !c}, {a, !b, c}, {a, b, !c}, {a, b, c}, eps, acc
    features = LDBAGraph.get_features(LDBATransition(0, 1, '!a | c', False, propositions), assignments)
    assert features == [1., 1., 1., 1., 0., 1., 0., 1., 0., 0.]
    features = LDBAGraph.get_features(
        LDBATransition(0, 1, '!a & b & !c | b & !c & a | c & a', True, propositions), assignments)
    assert features == [0., 0., 1., 0., 0., 1., 1., 1., 0., 1.]
