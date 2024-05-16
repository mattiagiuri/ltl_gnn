import torch
from torch import nn


class BatchedSequence:
    def __init__(self, seqs: list[list[tuple[int, int]]], device=None):
        self.batch_reach, self.batch_avoid, self.batch_lens = self.batch(seqs, device)

    @classmethod
    def batch(cls, seqs: list[list[tuple[int, int]]], device=None) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        reach_seqs = map(lambda seq: [s[0] for s in seq], seqs)
        avoid_seqs = map(lambda seq: [s[1] for s in seq], seqs)
        lens = [len(seq) for seq in seqs]
        padded_reach = nn.utils.rnn.pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in reach_seqs],
                                                 batch_first=True)
        padded_avoid = nn.utils.rnn.pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in avoid_seqs],
                                                 batch_first=True)
        return padded_reach.to(device), padded_avoid.to(device), torch.tensor(lens, dtype=torch.long)  # lens on CPU

    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Returns a sub-batch of the given sequences."""
        return self.batch_reach[index], self.batch_avoid[index], self.batch_lens[index]

    def all(self):
        return self.batch_reach, self.batch_avoid, self.batch_lens
