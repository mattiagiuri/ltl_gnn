VOCAB = {
    'PAD': 0,
    'EMPTY': 1,
    'NULL': 2,
    'AND': 3,
    'OR': 4,
}


def init_vocab(propositions: list[str]):
    for p in sorted(propositions):
        if p not in VOCAB:
            VOCAB[p] = len(VOCAB)
