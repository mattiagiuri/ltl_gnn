from typing import Any

import torch

import torch_ac
import numpy as np

from model.ltl.batched_sequence import BatchedSequence

VOCAB = {'PAD': 0}


def preprocess_obss(obss: list[dict[str, Any]], device=None) -> torch_ac.DictList:
    features = []
    seqs = []
    for obs in obss:
        features.append(obs["features"])
        seqs.append(list(reversed(obs["goal"])))
    return torch_ac.DictList({
        "features": preprocess_features(features, device=device),
        "seq": BatchedSequence([preprocess_sequence(seq) for seq in seqs], device=device),
        # "goal": torch.tensor([obs["goal_index"] for obs in obss], device=device, dtype=torch.long),
    })


def preprocess_features(features, device=None) -> torch.tensor:
    return torch.tensor(np.array(features), dtype=torch.float).to(device)


def preprocess_sequence(seq: list[tuple[str, str]]) -> list[tuple[int, int]]:
    return [(VOCAB.setdefault(s[0], len(VOCAB)), VOCAB.setdefault(s[1], len(VOCAB))) for s in seq]
