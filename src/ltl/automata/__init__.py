from contextlib import nullcontext
from typing import Optional

from utils import memory, timeit
from .rabinizer import run_rabinizer
from .ldba import LDBA, LDBATransition
from .ldba_graph import LDBAGraph


@memory.cache
def ltl2ldba(formula: str, propositions: list[str] = None, simplify_labels=True, print_time=False) -> LDBA:
    """Converts an LTL formula to an LDBA using the rabinizer tool."""
    from ltl.hoa import HOAParser
    with timeit(f'Converting LTL formula "{formula}" to LDBA') if print_time else nullcontext():
        hoa = run_rabinizer(formula)
    return HOAParser(formula, hoa, propositions, simplify_labels=simplify_labels).parse_hoa()


__all__ = ['LDBA', 'LDBATransition', 'LDBAGraph', 'ltl2ldba']
