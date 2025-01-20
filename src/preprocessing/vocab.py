from ltl.logic import Assignment

FIXED = ['PAD', 'EPSILON', 'NULL']
VOCAB = {k: i for i, k in enumerate(FIXED)}
assignment_vocab = {i: k for i, k in enumerate(FIXED)}
var_names = []


def init_vocab(assignments: list[Assignment]):
    for a in assignments:
        VOCAB[a.to_frozen()] = len(VOCAB)
        assignment_name = "&".join([x for x, v in a.mapping.items() if v])
        assignment_vocab[len(assignment_vocab)] = assignment_name if len(assignment_name) > 0 else "blank"


def init_vars(variables: list[str]):
    global var_names
    var_names += variables
    # var_names.append("blank")
