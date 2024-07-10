from typing import Any

import torch

import torch_ac
import numpy as np

from ltl.automata import LDBASequence
from ltl.logic import FrozenAssignment
from preprocessing.batched_ast_sequence import BatchedASTSequence
from preprocessing.assignment_ast import *


def preprocess_obss(obss: list[dict[str, Any]], device=None) -> torch_ac.DictList:
    features = []
    seqs = []
    for obs in obss:
        features.append(obs["features"])
        seqs.append(list(reversed(obs["goal"])))
    return torch_ac.DictList({
        "features": preprocess_features(features, device=device),
        "seq": BatchedASTSequence([preprocess_sequence(seq) for seq in seqs], device=device),
    })


def preprocess_features(features, device=None) -> torch.tensor:
    return torch.tensor(np.array(features), dtype=torch.float).to(device)


def preprocess_sequence(seq: LDBASequence) -> list[tuple[ASTNode, ASTNode]]:
    return [(preprocess_assignments(a), preprocess_assignments(b)) for a, b in seq]


def preprocess_assignments(assignments: frozenset[FrozenAssignment]) -> ASTNode:
    if len(assignments) == 0:
        return NullNode()
    if len(assignments) == 1:
        return preprocess_assignment(next(iter(assignments)))
    return OrNode([preprocess_assignment(a) for a in assignments])


def preprocess_assignment(assignment: FrozenAssignment) -> ASTNode:
    active = [t[0] for t in assignment if t[1]]
    if len(active) == 0:
        return EmptyNode()
    if len(active) == 1:
        return PropositionNode(active[0])
    return AndNode([PropositionNode(p) for p in active])
