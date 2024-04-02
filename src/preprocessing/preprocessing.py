import torch

import torch_ac
import numpy as np


def preprocess_obss(obss: list[dict[str, np.ndarray | str]], device=None) -> torch_ac.DictList:
    return torch_ac.DictList({
        "features": preprocess_features([obs["features"] for obs in obss], device=device),
        "goal": torch.LongTensor([obs["goal_index"] for obs in obss], device=device)
    })


def preprocess_features(features, device=None) -> torch.tensor:
    return torch.tensor(np.array(features), dtype=torch.float).to(device)
