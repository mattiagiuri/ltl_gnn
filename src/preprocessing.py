import torch

import src.torch_ac as torch_ac
import numpy as np


def preprocess_obss(obss: list[dict[str, np.ndarray | str]], device=None) -> torch_ac.DictList:
    return torch_ac.DictList({
        "features": preprocess_features([obs["features"] for obs in obss], device=device),
        "ltl": preprocess_ltl([obs["text"] for obs in obss], device=device)
    })


def preprocess_features(features, device=None) -> torch.tensor:
    return torch.tensor(np.array(features), dtype=torch.float).to(device)


def preprocess_ltl(ltl: list[str], device=None) -> torch.tensor:
    def text_to_index(text: str | tuple) -> int:
        if isinstance(text, tuple):
            text = text[1]
        return {
            'g': 0,
            'b': 1,
            'y': 2,
            'm': 3,
            'True': 4,
        }[text]

    indices = [text_to_index(t) for t in ltl]
    return torch.LongTensor(indices).to(device)
