import enum

from graphviz import Source

from ltl.automata import LDBA, ltl2ldba, ltl2nba
from ltl.logic import Assignment
from sequence.search import ExhaustiveSearch
import os


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
    print(dot)
    s = Source(dot, filename=filename, format=fmt)
    s.render(view=view, cleanup=True)


# @memory.cache
def construct_ldba_flatworld(formula: str, simplify_labels: bool = False, prune: bool = True, ldba: bool = True) -> LDBA:
    fun = ltl2ldba if ldba else ltl2nba
    ldba = fun(formula, simplify_labels=simplify_labels)
    print('Constructed LDBA.')
    # assert ldba.check_valid()
    # print('Checked valid.')
    if prune:
        # ldba.prune(Assignment.zero_or_one_propositions(set(ldba.propositions)))
        props = {'red', 'magenta', 'blue', 'green', 'aqua', 'yellow', 'orange'}
        ldba.prune([
            Assignment.where('red', propositions=props),
            Assignment.where('magenta', propositions=props),
            Assignment.where('red', 'magenta', propositions=props),
            Assignment.where('blue', propositions=props),
            Assignment.where('green', propositions=props),
            Assignment.where('aqua', propositions=props),
            Assignment.where('blue', 'green', propositions=props),
            Assignment.where('green', 'aqua', propositions=props),
            Assignment.where('blue', 'aqua', propositions=props),
            Assignment.where('blue', 'green', 'aqua', propositions=props),
            Assignment.where('yellow', propositions=props),
            Assignment.where('orange', propositions=props),
            Assignment.zero_propositions(props),
        ])
        print('Pruned impossible transitions.')
    ldba.complete_sink_state()
    print('Added sink state.')
    ldba.compute_sccs()
    return ldba


def construct_ldba_chessworld(formula: str, simplify_labels: bool = False, prune: bool = True, ldba: bool = True) -> LDBA:
    fun = ltl2ldba if ldba else ltl2nba
    ldba = fun(formula, simplify_labels=simplify_labels)
    print('Constructed LDBA.')
    # assert ldba.check_valid()
    # print('Checked valid.')
    if prune:
        # ldba.prune(Assignment.zero_or_one_propositions(set(ldba.propositions)))
        props = {'queen', 'rook', 'bishop', 'knight', 'pawn'}
        ldba.prune([
            Assignment.where('queen', propositions=props),
            Assignment.where('rook', propositions=props),
            Assignment.where('knight', propositions=props),
            Assignment.where('bishop', propositions=props),
            Assignment.where('pawn', propositions=props),
            Assignment.where('queen', 'rook', propositions=props),
            Assignment.where('queen', 'bishop', propositions=props),
            Assignment.where('queen', 'pawn', 'bishop', propositions=props),
            Assignment.where('queen', 'rook', 'pawn', propositions=props),
            # Assignment.where('rook', 'pawn', propositions=props),
            Assignment.where('rook', 'knight', propositions=props),
            Assignment.where('rook', 'bishop', propositions=props),
            Assignment.where('knight', 'bishop', propositions=props),
            Assignment.zero_propositions(props),
        ])
        print('Pruned impossible transitions.')
    ldba.complete_sink_state()
    print('Added sink state.')
    ldba.compute_sccs()
    return ldba


def get_ldba_transitions(ldba):
    pass


if __name__ == '__main__':
    # os.chdir("..")
    # os.chdir("..")

    # props = {'red', 'magenta', 'blue', 'green', 'aqua', 'yellow', 'orange'}
    props = {'queen', 'rook', 'bishop', 'knight', 'pawn'}

    f = '(! (queen | rook) U bishop) & (F knight)'

    ldba = construct_ldba_chessworld(f, simplify_labels=True, prune=True, ldba=True)
    # print(ldba.state_to_transitions)
    # print(ldba.state_to_incoming_transitions)
    print(f'Finite: {ldba.is_finite_specification()}')
    # draw_ldba(ldba, fmt='png', positive_label=True, self_loops=True)
    search = ExhaustiveSearch(None, props, num_loops=1)
    seqs = search.all_sequences(ldba, ldba.initial_state)
    # print(seqs)
    for i in seqs:
        print(i)
    print(len(seqs))
