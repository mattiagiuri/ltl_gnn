from ltl.logic import Assignment

FIXED = ['PAD', 'EPSILON', 'NULL']
VOCAB = {k: i for i, k in enumerate(FIXED)}
assignment_vocab = {i: k for i, k in enumerate(FIXED)}
var_names = []


def init_vocab(assignments: list[Assignment]):
    for a in assignments:
        VOCAB[a.to_frozen()] = len(VOCAB)
        assignment_vocab[len(assignment_vocab)] = "&".join([x for x, v in a.mapping.items() if v])


def init_vars(variables: list[str]):
    global var_names
    var_names += variables
