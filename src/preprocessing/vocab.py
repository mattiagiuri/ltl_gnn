FIXED = ['PAD', 'EPSILON', 'EMPTY', 'NULL', 'AND', 'OR']
VOCAB = {k: i for i, k in enumerate(FIXED)}


def init_vocab(propositions: list[str]):
    for p in sorted(propositions):
        if p not in VOCAB:
            VOCAB[p] = len(VOCAB)
