import functools
from typing import Optional

from .rabinizer import run_rabinizer
from .ldba import LDBA, LDBATransition


@functools.cache
def ltl2ldba(formula: str, propositions: Optional[frozenset[str]] = None, simplify_labels=True) -> LDBA:
    """Converts an LTL formula to an LDBA using the rabinizer tool."""
    from ltl.hoa import HOAParser
    hoa = run_rabinizer(formula)
    return HOAParser(hoa, propositions, simplify_labels=simplify_labels).parse_hoa()


__all__ = ['LDBA', 'LDBATransition', 'ltl2ldba']
