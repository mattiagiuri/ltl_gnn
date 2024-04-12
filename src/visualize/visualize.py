import enum

import spot
from graphviz import Source

from ltl.automata import LDBA, ltl2ldba
from ltl.hoa import HOAWriter
from model.ltl import TransitionGraph


class Color(enum.Enum):
    SINK = 'tomato'
    ACCEPTING = 'lightskyblue'

    def __str__(self):
        return self.value


def draw_ldba(ldba: LDBA, filename='ldba', fmt='pdf', view=True) -> None:
    """Draw an LDBA as a graph using Graphviz."""
    hoa = HOAWriter(ldba).get_hoa()
    aut = spot.automaton(hoa)
    dot = aut.to_str('dot')
    if ldba.has_sink_state():
        dot = insert_dot_line(dot, f'{ldba.sink_state} [color="{Color.SINK}", style="filled"]')
    s = Source(dot, filename=filename, format=fmt)
    s.render(view=view, cleanup=True)


def insert_dot_line(dot: str, line: str) -> str:
    dot = dot.split('\n')
    dot.insert(-2, line)
    return '\n'.join(dot)


def draw_transition_graph(tg: TransitionGraph, filename='transition_graph', fmt='pdf', view=True) -> None:
    dot = 'digraph "" {\n'
    dot += 'rankdir=BT\n'
    for i, label in tg.info.labels.items():
        dot += f'{i} [label="{label if label is not None else "eps"}"'
        if i in tg.info.accepting_transitions:
            dot += f' color="{Color.ACCEPTING}", style="filled"'
        elif i in tg.info.epsilon_transitions:
            dot += ' style="dashed"'
        elif i in tg.info.sink_transitions:
            dot += f' color="{Color.SINK}", style="filled"'
        dot += ']\n'
    for edge in tg.edge_index.t().tolist():
        dot += f'{edge[0]} -> {edge[1]}\n'
    dot += '}'
    s = Source(dot, filename=filename, format=fmt)
    s.render(view=view, cleanup=True)


if __name__ == '__main__':
    # formula = '(!a U (b & (!c U d)))'
    # formula = '(F(a&b) | F(a & XFc)) & G!d'
    # formula = '(F(a&b) | F(a & XFb))'
    # formula = 'F((a&b)&FGb)'
    # formula = '(Fc) & (G(a => F b))'
    formula = '(FGa | FGb) & G!c'
    ldba = ltl2ldba(formula)
    ldba.complete_sink_state()
    draw_ldba(ldba, fmt='png')
    tg = TransitionGraph.from_ldba(ldba)
    draw_transition_graph(tg, fmt='png')
