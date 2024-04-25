import enum

from graphviz import Source

from ltl.automata import LDBA, ltl2ldba
from ltl.logic import Assignment
from model.ltl import TransitionGraph


class Color(enum.Enum):
    SINK = 'tomato'
    ACCEPTING = 'lightskyblue'

    def __str__(self):
        return self.value


def draw_ldba(ldba: LDBA, filename='ldba', fmt='pdf', view=True, positive_label=False) -> None:
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
            dot += f'{state} -> {transition.target} [label="{transition.label if not positive_label else transition.positive_label}"'
            if transition.accepting:
                dot += f' color="{Color.ACCEPTING}"'
            dot += ']\n'
    dot += '}'
    s = Source(dot, filename=filename, format=fmt)
    s.render(view=view, cleanup=True)


def draw_transition_graph(
        tg: TransitionGraph,
        filename='transition_graph',
        fmt='pdf',
        view=True,
        positive_label=False,
        features=False
) -> None:
    """Draw a transition graph as a graph using Graphviz. Uses labels by default, but can also use feature vectors."""
    if positive_label and features:
        raise ValueError('Can only visualize one of positive_label and features.')
    dot = 'digraph "" {\n'
    dot += 'rankdir=BT\n'
    if positive_label:
        nodes = tg.info.positive_labels.items()
    elif features:
        nodes = enumerate(tg.x.tolist())
    else:
        nodes = tg.info.labels.items()
    for i, label in nodes:
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
    formula = '(!a U (b & (!c U d)))'
    # formula = 'F (g & G!r) & G!b'
    # formula = '(!a U (b & (!c U d))) & (!e U (f & (!g U h)))'
    # formula = '!a U (b & (!c U (d & (!e U f))))'
    # formula = '(F(a&b) | F(a & XFc)) & G!d'
    # formula = '(F(a&b) | F(a & XFb))'
    # formula = 'F((a&b)&FGb)'
    # formula = '(FGa | FGb) & G!c'
    # formula = 'GFa & GFb & G!c'
    # formula = '!r U g'
    prune = True


    # ldba = ltl2ldba(formula, propositions=frozenset({'green', 'red', 'yellow', 'blue'}), simplify_labels=True)
    ldba = ltl2ldba(formula, simplify_labels=True)
    assert ldba.check_valid()
    print('Constructed LDBA.')
    ldba.complete_sink_state()
    print('Added sink state.')
    if prune:
        ldba.prune_impossible_transitions(Assignment.more_than_one_true_proposition(set(ldba.propositions)))
        print('Pruned impossible transitions.')
    draw_ldba(ldba, fmt='pdf', positive_label=True)
    # tg = TransitionGraph.from_ldba(ldba)
    # print('Constructed transition graph.')
    # draw_transition_graph(tg, fmt='pdf', positive_label=True, features=False)
