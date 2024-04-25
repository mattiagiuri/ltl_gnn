from typing import Any

import torch

import torch_ac
import numpy as np

from model.ltl.batched_transition_graph import BatchedTransitionGraph


def preprocess_obss(obss: list[dict[str, Any]], device=None) -> torch_ac.DictList:
    features = []
    tgs = []
    active_transitions = []
    for obs in obss:
        features.append(obs["features"])
        tgs.append(obs["transition_graph"])
        active_transitions.append(obs["active_transitions"])
    return torch_ac.DictList({
        "features": preprocess_features(features, device=device),
        "transition_graph": BatchedTransitionGraph(tgs, device=device),
        # "goal": torch.tensor([obs["goal_index"] for obs in obss], device=device, dtype=torch.long),
    })


def preprocess_features(features, device=None) -> torch.tensor:
    return torch.tensor(np.array(features), dtype=torch.float).to(device)
