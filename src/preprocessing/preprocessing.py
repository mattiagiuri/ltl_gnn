from typing import Any

import torch

import torch_ac
import numpy as np

from model.ltl.batched_graph import BatchedGraph


def preprocess_obss(obss: list[dict[str, Any]], device=None) -> torch_ac.DictList:
    features = []
    pos_graphs = []
    neg_graphs = []
    for obs in obss:
        features.append(obs["features"])
        pos_graphs.append(obs["pos_graph"])
        neg_graphs.append(obs["neg_graph"])
    return torch_ac.DictList({
        "features": preprocess_features(features, device=device),
        "pos_graph": BatchedGraph(pos_graphs, device=device),
        "neg_graph": BatchedGraph(neg_graphs, device=device),
        # "goal": torch.tensor([obs["goal_index"] for obs in obss], device=device, dtype=torch.long),
    })


def preprocess_features(features, device=None) -> torch.tensor:
    return torch.tensor(np.array(features), dtype=torch.float).to(device)
