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
    f = 'F (blue & F green) & G(!magenta & !yellow)'

    ldba = construct_ldba(f, simplify_labels=False, prune=True, ldba=True)
    print(f'Finite: {ldba.is_finite_specification()}')
    draw_ldba(ldba, fmt='png', positive_label=True, self_loops=True)
    search = ExhaustiveSearch(None, num_loops=1)
    seqs = search.all_sequences(ldba, ldba.initial_state)
    print(seqs)
    print(len(seqs))
