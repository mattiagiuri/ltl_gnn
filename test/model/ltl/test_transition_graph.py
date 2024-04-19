import pytest
import torch

from ltl.automata import LDBA, LDBATransition
from ltl.logic import Assignment
from model.ltl import TransitionGraph


def assert_edge_index_equal(edge_index1, edge_index2):
    assert edge_index1.shape == edge_index2.shape
    edges1 = set(map(tuple, edge_index1.t().tolist()))
    for edge2 in edge_index2.t().tolist():
        assert tuple(edge2) in edges1


def test_from_ldba1():
    # formula: F(a&b) | F(a & XFb)
    ldba = LDBA({'a', 'b'})
    ldba.add_state(0, initial=True)
    ldba.add_state(1)
    ldba.add_state(2)
    ldba.add_transition(0, 0, '!a', False)
    ldba.add_transition(0, 1, 'a & b', True)
    ldba.add_transition(0, 2, 'a & !b', False)
    ldba.add_transition(1, 1, 't', True)
    ldba.add_transition(2, 2, '!b', False)
    ldba.add_transition(2, 1, 'b', True)
    ldba.complete_sink_state()
    assert ldba.check_valid()

    tg = TransitionGraph.from_ldba(ldba)
    assert tg.validate()
    assert tg.x.shape[0] == 6
    assert tg.edge_index.max() == 5
    expected_features = torch.tensor([
        [1., 1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 1.],
        [0., 0., 1., 0., 0., 0.],
        [1., 1., 1., 1., 0., 1.],
        [0., 1., 0., 1., 0., 1.],
        [1., 0., 1., 0., 0., 0.]
    ])
    assert (tg.x == expected_features).all()
    expected_labels = {0: '!a', 1: 'a & b', 2: 'a & !b', 3: 't', 5: '!b', 4: 'b'}
    assert tg.info.labels == expected_labels
    expected_edge_index = torch.tensor([
        [0, 1, 2, 3, 3, 3, 5, 5, 4, 4],
        [0, 0, 0, 1, 3, 4, 2, 5, 2, 5]
    ])
    assert_edge_index_equal(expected_edge_index, tg.edge_index)
    assert tg.info.accepting_transitions == {1, 3, 4}
    assert tg.info.epsilon_transitions == set()
    assert tg.info.sink_transitions == set()


def test_from_ldba2():
    # formula: F((a & b) & FGb)
    ldba = LDBA({'a', 'b'})
    ldba.add_state(0, initial=True)
    ldba.add_state(1)
    ldba.add_state(2)
    ldba.add_transition(0, 0, '!(a & b)', False)
    ldba.add_transition(0, 1, 'a & b', False)
    ldba.add_transition(1, 1, 't', False)
    ldba.add_transition(1, 2, None, False)
    ldba.add_transition(2, 2, 'b', True)
    assert ldba.check_valid()
    with pytest.raises(ValueError):
        TransitionGraph.from_ldba(ldba)
    ldba.complete_sink_state()
    assert ldba.check_valid()

    tg = TransitionGraph.from_ldba(ldba)
    assert tg.validate()
    assert tg.x.shape[0] == 7
    assert tg.edge_index.max() == 6
    expected_features = torch.tensor([
        [1., 1., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [1., 1., 1., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 1., 0., 1., 0., 1.],
        [1., 0., 1., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0.]
    ])
    assert (tg.x == expected_features).all()
    expected_labels = {0: '!a | !b', 1: 'a & b', 2: 't', 3: None, 4: 'b', 5: '!b', 6: 't'}
    assert tg.info.labels == expected_labels
    expected_edge_index = torch.tensor([[0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
                                        [0, 0, 1, 2, 1, 2, 3, 4, 3, 4, 6, 5]], dtype=torch.long)
    assert_edge_index_equal(expected_edge_index, tg.edge_index)
    assert tg.info.accepting_transitions == {4}
    assert tg.info.epsilon_transitions == {3}
    assert tg.info.sink_transitions == {5, 6}


def test_from_ldba3():
    # formula: FGa | FGb
    ldba = LDBA({'a', 'b'})
    ldba.add_state(0, initial=True)
    ldba.add_state(1)
    ldba.add_state(2)
    ldba.add_transition(0, 0, 't', False)
    ldba.add_transition(0, 1, None, False)
    ldba.add_transition(1, 1, 'a', True)
    ldba.add_transition(0, 2, None, False)
    ldba.add_transition(2, 2, 'b', True)
    assert ldba.check_valid()
    ldba.complete_sink_state()
    assert ldba.check_valid()

    tg = TransitionGraph.from_ldba(ldba)
    assert tg.validate()
    assert tg.x.shape[0] == 8
    assert tg.edge_index.max() == 7
    expected_features = torch.tensor([
        [1., 1., 1., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 1., 1., 0., 1.],
        [1., 1., 0., 0., 0., 0.],
        [0., 1., 0., 1., 0., 1.],
        [1., 0., 1., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0.]
    ])
    assert (tg.x == expected_features).all()
    expected_labels = {0: 't', 1: None, 2: None, 3: 'a', 4: '!a', 5: 'b', 6: '!b', 7: 't'}
    assert tg.info.labels == expected_labels
    expected_edge_index = torch.tensor([[0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7],
                                        [0, 0, 0, 1, 3, 1, 3, 2, 5, 2, 5, 7, 4, 6]], dtype=torch.long)
    assert_edge_index_equal(expected_edge_index, tg.edge_index)
    assert tg.info.accepting_transitions == {3, 5}
    assert tg.info.epsilon_transitions == {1, 2}
    assert tg.info.sink_transitions == {4, 6, 7}


def test_from_pruned_ldba():
    ldba = LDBA({'a', 'b', 'c'})
    ldba.add_state(0, initial=True)
    ldba.add_state(1)
    ldba.add_state(2)
    ldba.add_transition(0, 0, '!a & !c', False)
    ldba.add_transition(0, 0, 'a & b & !c', True)
    ldba.add_transition(0, 1, 'a & !b & !c', False)
    ldba.add_transition(0, 2, 'c', False)
    ldba.add_transition(1, 1, '!b & !c', False)
    ldba.add_transition(1, 0, 'b & !c', True)
    ldba.add_transition(1, 2, 'c', False)
    ldba.add_transition(2, 2, 't', False)
    ldba.complete_sink_state()
    assert not ldba.has_sink_state()
    ldba.sink_state = 2
    assert ldba.check_valid()
    more_than_one_proposition = {a.to_frozen() for a in Assignment.all_possible_assignments(ldba.propositions)
                                 if len([v for v in a.values() if v]) > 1}
    ldba.prune_impossible_transitions(more_than_one_proposition)

    tg = TransitionGraph.from_ldba(ldba)
    assert tg.validate()
    assert tg.x.shape[0] == 7
    assert tg.edge_index.max() == 6
    expected_features = torch.tensor([
        # {}, c, b, a
        [1., 0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0., 1.],
        [0., 0., 0., 1., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [1., 0., 0., 1., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0.]
    ])
    assert (tg.x == expected_features).all()
    expected_labels = {0: '!a & !c', 2: 'a & !b & !c', 3: 'c & !a & !b', 4: '!b & !c', 1: 'b & !a & !c',
                       5: 'c & !a & !b', 6: '(!a & !b) | (!a & !c) | (!b & !c)'}
    assert tg.info.labels == expected_labels
    expected_edge_index = torch.tensor([[0, 0, 2, 2, 4, 4, 1, 1, 3, 3, 5, 5, 6, 6, 6],
                                        [0, 1, 0, 1, 2, 4, 4, 2, 0, 1, 4, 2, 5, 3, 6]])
    assert_edge_index_equal(expected_edge_index, tg.edge_index)
    assert tg.info.accepting_transitions == {1}
    assert tg.info.epsilon_transitions == set()
    assert tg.info.sink_transitions == {3, 5, 6}


def test_get_features():
    propositions = ('a', 'b')
    assignments = Assignment.all_possible_assignments(propositions)
    # components: {!a, !b}, {!a, b}, {a, !b}, {a, b}, eps, acc
    features = TransitionGraph.get_features(LDBATransition(0, 1, 'a & b', False, propositions), assignments)
    assert features.tolist() == [0., 0., 0., 1., 0., 0.]
    features = TransitionGraph.get_features(LDBATransition(0, 1, 'a', True, propositions), assignments)
    assert features.tolist() == [0., 0., 1., 1., 0., 1.]
    features = TransitionGraph.get_features(LDBATransition(0, 1, None, False, propositions), assignments)
    assert features.tolist() == [0., 0., 0., 0., 1., 0.]

    propositions = ('a', 'b', 'c')
    assignments = Assignment.all_possible_assignments(propositions)
    # components:
    # {!a, !b, !c}, {!a, !b, c}, {!a, b, !c}, {!a, b, c}, {a, !b, !c}, {a, !b, c}, {a, b, !c}, {a, b, c}, eps, acc
    features = TransitionGraph.get_features(LDBATransition(0, 1, '!a | c', False, propositions), assignments)
    assert features.tolist() == [1., 1., 1., 1., 0., 1., 0., 1., 0., 0.]
    features = TransitionGraph.get_features(
        LDBATransition(0, 1, '!a & b & !c | b & !c & a | c & a', True, propositions), assignments)
    assert features.tolist() == [0., 0., 1., 0., 0., 1., 1., 1., 0., 1.]
