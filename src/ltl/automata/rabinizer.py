import subprocess

from ltl.automata.utils import draw_ldba
from ltl.hoa import HOAParser

RABINIZER_PATH = 'rabinizer4/bin/ltl2ldba'


def ltl2ldba(formula: str) -> str:
    """Convert an LTL formula to a LDBA in the HOA format."""
    # -p: parallel processing
    # -d: construct a non-generalised Buechi automaton
    # -e: keep generated epsilon transitions
    command = [RABINIZER_PATH, '-i', formula, '-p', '-d', '-e']
    run = subprocess.run(command, capture_output=True, text=True)
    if run.stderr != '':
        raise RuntimeError(f'Rabinizer call `{" ".join(command)}` resulted in an error.\nError: {run.stderr}.')
    return run.stdout


if __name__ == '__main__':
    # formula = '(!a U (b & (!c U d)))'
    # formula = '(F(a&b) | F(a & XFc)) & G!d'
    # formula = '(F(a&b) | F(a & XFb))'
    formula = 'F((a&b)&FGb)'
    # formula = '(Fc) & (G(a => F b))'
    # formula = 'FGa | FGb'
    hoa = ltl2ldba(formula)
    with open('ldba.hoa', 'w') as file:
        file.write(hoa)
    ldba = HOAParser(hoa).parse_hoa()
    ldba.complete_sink_state()
    draw_ldba(ldba, fmt='png')
