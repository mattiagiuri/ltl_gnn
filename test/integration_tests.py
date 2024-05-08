from ltl.automata import ltl2ldba
from ltl.logic import Assignment
from model.ltl import LDBAGraph


def test_ldba_graph_from_formula():
    formula = '(!a U (b & (!c U d))) & (!e U (f & (!g U h)))'
    ldba = ltl2ldba(formula, simplify_labels=False)
    all_possible = Assignment.all_possible_assignments(ldba.propositions)
    assert ldba.check_valid()
    ldba.complete_sink_state()
    more_than_one_proposition = {a.to_frozen() for a in all_possible if len([v for v in a.values() if v]) > 1}
    ldba.prune_impossible_transitions(more_than_one_proposition)
    assert ldba.num_states == 17
    assert ldba.num_transitions == 69
    ldba.compute_sccs()
    pos, neg = LDBAGraph.from_ldba(ldba, ldba.initial_state)
    assert pos.num_nodes == 38
    assert pos.num_edges == 73
    assert neg.num_nodes == 47
    assert neg.num_edges == 87
