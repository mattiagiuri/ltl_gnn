import enum

from graphviz import Source

from ltl.automata import LDBA, ltl2ldba, ltl2nba
from ltl.logic import Assignment
from sequence.search import ExhaustiveSearch


class Color(enum.Enum):
    SINK = 'tomato'
    ACCEPTING = 'lightskyblue'
    ROOT = '#ffcc00'

    def __str__(self):
        return self.value


def draw_ldba(ldba: LDBA, filename='ldba', fmt='pdf', view=True, positive_label=False, self_loops=True) -> None:
    """Draw an LDBA as a graph using Graphviz."""
    dot = 'digraph "" {\n'
    dot += 'rankdir=LR\n'
    dot += 'labelloc="t"\n'
    dot += 'node [shape="circle"]\n'
    dot += 'I [label="", style=invis, width=0]\n'
    dot += f'I -> {ldba.initial_state}\n'
    for state, transitions in ldba.state_to_transitions.items():
        dot += f'{state} [label="{state}"'
        if state == ldba.sink_state:
            dot += f' color="{Color.SINK}" style="filled"'
        dot += ']\n'
        for transition in transitions:
            if not self_loops and transition.target == state:
                continue
            dot += f'{state} -> {transition.target} [label="{transition.label if not positive_label else transition.positive_label}"'
            if transition.accepting:
                dot += f' color="{Color.ACCEPTING}"'
            dot += ']\n'
    dot += '}'
    s = Source(dot, filename=filename, format=fmt)
    s.render(view=view, cleanup=True)


# @memory.cache
def construct_ldba(formula: str, simplify_labels: bool = False, prune: bool = True, ldba: bool = True) -> LDBA:
    fun = ltl2ldba if ldba else ltl2nba
    ldba = fun(formula, simplify_labels=simplify_labels)
    print('Constructed LDBA.')
    # assert ldba.check_valid()
    # print('Checked valid.')
    if prune:
        ldba.prune(Assignment.zero_or_one_propositions(set(ldba.propositions)))
        # ldba.prune([
        #     Assignment(dict(a=False, r=False)),
        #     Assignment(dict(a=False, r=True)),
        #     Assignment(dict(a=True, r=False)),
        #     Assignment(dict(a=True, r=True)),
        # ])
        print('Pruned impossible transitions.')
    ldba.complete_sink_state()
    print('Added sink state.')
    ldba.compute_sccs()
    return ldba


if __name__ == '__main__':
    # f = '!a U b'
    # f = 'FGa'
    # f = 'GF a & GF b & G (s => F g)'
    f = 'F (green & (!yellow U blue)) & G (!magenta)'
    # f = '(F (g & F (f & F (d & F (g & F (a & F (h & F (b & F (j & F (f & F (g & F (i & F (b & F (c & F (f & F h)))))))))))))))'
    # f = '!a U (b & (!c U d))'
    # f = '!a U (b & (!c U (d & (!e U f))))'
    # f = '(s => F r) U g'
    # f = 'GF a & GF b & G (c => (!d U a))'
    # f = 'F (a | b)'
    # f = 'F (g & G!r) & G!b'
    # f = '(!a U (b & (!c U d))) & (!e U (f & (!g U h))) & (!i U (j & (!k U l)))'
    # f = 'F (b & F (d & F f)) & F (h & F (j & F l))'
    # f = '(!a U (b & (!c U (d & (!e U f))))) & (!g U (h & (!i U j)))'
    # f = '(F(a&b) | F(a & XFc)) & G!d'
    # f = '(F(a&b) | F(a & XFb))'
    # f = 'F((a&b)&FGb)'
    # f = 'GFa & GFb & G!c'
    # f = 'F a | F (b & F c)'
    # f = 'GFa & GFb & G (a => F g)'
    # f = '!r U g'
    # f = 'F (g & G!r)'
    # f = 'F g & G!r'
    # f = 'FGr'
    # f = '(GF b) & (GF r) & (GF a)'  # make sure to set prune correctly.
    # f = 'r & X ((g & X y) | y)'
    # f = 'GF(r & Xb)'
    # f = 'GF(r & Fb) | GF(r & F(g & Fb))'
    # f = 'a & XGFa | b & XGFb'
    # f = 'F(g&Fb)'
    # f = 'r & !(Xg) & G(!y | (X g))'
    # f = 'F ((blue | yellow) & F red)'
    # f = 'GF a | G (!b | F c)'
    # f = '(!h U (k & (!b U (c & (!i U e)))))'
    # f = '(F (a & (F d))) | (F (b & (F (c & (F d)))))'
    # f = '(!a U (k & (!i U (c & (!b U (h & (!j U (g & (!f U (d & (!l U e)))))))))))'
    # f = 'F (r & (a | b))'
    # f = '!(a | b) U c'
    # f = 'F (a & !r)'
    # f = '(F a) U b'
    # f = '(!e U (i & (!j U d))) & (!c U (h & (!b U f))) & (!g U (a & (!l U k)))'
    # f = 'F a & (!h U a)'
    # f = 'F (i & F (b & F d)) & F (e & F ((k | f) & F c)) & F (j & F (e & F g))'

    ldba = construct_ldba(f, simplify_labels=False, prune=True, ldba=True)
    print(f'Finite: {ldba.is_finite_specification()}')
    draw_ldba(ldba, fmt='png', positive_label=True, self_loops=True)
    seqs = ExhaustiveSearch.all_sequences(ldba, ldba.initial_state)
    print(seqs)
    print(len(seqs))
