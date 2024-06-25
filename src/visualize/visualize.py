import enum
from pprint import pprint

from graphviz import Source

from ltl.automata import LDBA, ltl2ldba
from ltl.automata import LDBAGraph
from ltl.logic import Assignment
from utils import memory, timeit


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


def draw_old_ldba_graph(
        g: LDBAGraph,
        filename='ldba_graph',
        fmt='pdf',
        view=True,
        features=False
) -> None:
    """Draw an LDBA graph as a graph using Graphviz. Uses labels by default, but can also use feature vectors."""
    dot = 'digraph "" {\n'
    dot += 'rankdir=BT\n'
    if features:
        nodes = enumerate(g.x.tolist())
    else:
        nodes = g.labels.items()
    for i, label in nodes:
        dot += f'{i} [label="{label}"'
        dot += ']\n'
    for edge in g.edge_index.t().tolist():
        if not edge:
            continue
        dot += f'{edge[0]} -> {edge[1]}\n'
    dot += '}'
    s = Source(dot, filename=filename, format=fmt)
    s.render(view=view, cleanup=True)


def draw_ldba_graph(
        g: LDBAGraph,
        filename='ldba_graph',
        fmt='pdf',
        view=True,
) -> None:
    dot = 'digraph "" {\n'
    dot += 'rankdir=BT\n'
    for node in graph.nodes:
        dot += f'{node.id} [label="({node.pos_label}, {node.neg_label if node.neg_label else "-"})"'
        if node.id in g.root_nodes:
            dot += f' color="{Color.ROOT}" style="filled"'
        dot += ']\n'
    for edge in g.edges:
        dot += f'{edge[0]} -> {edge[1]}\n'
    dot += '}'
    s = Source(dot, filename=filename, format=fmt)
    s.render(view=view, cleanup=True)


# @memory.cache
def construct_ldba(formula: str, simplify_labels: bool = False, prune: bool = True) -> LDBA:
    ldba = ltl2ldba(formula, simplify_labels=simplify_labels)
    print('Constructed LDBA.')
    # assert ldba.check_valid()
    # print('Checked valid.')
    if prune:
        ldba.prune(Assignment.zero_or_one_propositions(set(ldba.propositions)))
        print('Pruned impossible transitions.')
    ldba.complete_sink_state()
    print('Added sink state.')
    ldba.compute_sccs()
    return ldba


if __name__ == '__main__':
    # f = '!a U b'
    # f = 'FGa'
    # f = 'GFa & GFb & G (signal => F g)'
    # f = '(F (g & F (f & F (d & F (g & F (a & F (h & F (b & F (j & F (f & F (g & F (i & F (b & F (c & F (f & F h)))))))))))))))'
    # f = '!a U (b & (!c U d))'
    # f = '!a U (b & (!c U (d & (!e U f))))'
    f = 'GF a & GF b & G (c => (!d U a))'  # TODO: here in state 4 => need to prune based on loops rather than just sequences
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
    # f = 'GFa & GFb & G (!a | F g)'
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

    ldba = construct_ldba(f, simplify_labels=False, prune=True)
    draw_ldba(ldba, fmt='png', positive_label=True, self_loops=True)
    graph = LDBAGraph.from_ldba(ldba, 0)  # try other states!
    draw_ldba_graph(graph, fmt='png')  # Crucial: prune paths that overlap with accepting paths!
    # draw_ldba_graph(neg, fmt='pdf')
    # print(pos.root_assignments)
