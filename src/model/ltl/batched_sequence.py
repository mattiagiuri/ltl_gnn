import functools

import torch
from torch import nn
from torch_geometric.data import Data


class BatchedSequence:

    VOCAB = {'PAD': 0}

    def __init__(self, seqs: list[list[str]], device=None):
        self.batch_seq, self.batch_lens = self.batch(seqs, device)
        # print(self.VOCAB)

    @classmethod
    def batch(cls, seqs: list[list[str]], device=None) -> tuple[torch.tensor, torch.tensor]:
        seqs = [[cls.VOCAB.setdefault(s, len(cls.VOCAB)) for s in seq] for seq in seqs]
        lens = [len(seq) for seq in seqs]
        padded = nn.utils.rnn.pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in seqs], batch_first=True)
        return padded.to(device), torch.tensor(lens, dtype=torch.long).to(device)

    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor]:
        """Returns a sub-batch of the given sequences."""
        return self.batch_seq[index], self.batch_lens[index]

    def all(self):
        return self.batch_seq, self.batch_lens
