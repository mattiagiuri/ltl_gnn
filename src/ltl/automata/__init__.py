import functools

from .rabinizer import run_rabinizer
from .ldba import LDBA, LDBATransition


@functools.cache
def ltl2ldba(formula: str) -> LDBA:
    """Converts an LTL formula to an LDBA using the rabinizer tool."""
    from ltl.hoa import HOAParser
    hoa = run_rabinizer(formula)
    return HOAParser(hoa).parse_hoa()


__all__ = ['LDBA', 'LDBATransition', 'ltl2ldba']
