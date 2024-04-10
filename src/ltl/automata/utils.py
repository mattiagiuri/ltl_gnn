import spot
from graphviz import Source

from ltl.automata.ldba import LDBA
from ltl.hoa.hoa_parser import HOAParser
from ltl.hoa.hoa_writer import HOAWriter


def render_ldba(ldba: LDBA, filename='tmp_ldba', fmt='pdf', view=True) -> None:
    """Render an LDBA as a graph using Graphviz."""
    hoa = HOAWriter(ldba).get_hoa()
    aut = spot.automaton(hoa)
    dot = aut.to_str('dot')
    s = Source(dot, filename=filename, format=fmt)
    s.render(view=view, cleanup=True)


if __name__ == '__main__':
    with open('ldba.hoa', 'r') as file:
        hoa_text = file.read()
    ldba = HOAParser(hoa_text).parse_hoa()
    assert ldba.check_valid()
    render_ldba(ldba)
