from ltl.automata import ltl2ldba
from ltl.logic import Assignment
from model.ltl import TransitionGraph


def test_transition_graph_from_formula():
    formula = '(!a U (b & (!c U d))) & (!e U (f & (!g U h)))'
    ldba = ltl2ldba(formula, simplify_labels=False)
    all_possible = Assignment.all_possible_assignments(ldba.propositions)
    assert ldba.check_valid()
    ldba.complete_sink_state()
    more_than_one_proposition = {a.to_frozen() for a in all_possible if len([v for v in a.values() if v]) > 1}
    ldba.prune_impossible_transitions(more_than_one_proposition)
    assert ldba.num_states == 17
    assert ldba.num_transitions == 69
    tg = TransitionGraph.from_ldba(ldba)
    assert tg.num_nodes == 69
    assert tg.num_edges == 239
