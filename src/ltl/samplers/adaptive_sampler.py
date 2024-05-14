import random

import numpy as np
import torch

from ltl.samplers import LTLSampler


class AdaptiveSampler(LTLSampler):
    def __init__(self, propositions: list[str]):
        super().__init__(propositions)
        self.temperature = 0.5
        self.task_returns: dict[str, float] = {f'F {a}': 0.0 for a in propositions}
        self.stage = 0
        self.threshold = 0.8

    def sample(self) -> str:
        tasks, probs = self.compute_sampling_prob()
        return np.random.choice(tasks, p=probs)

    def compute_sampling_prob(self) -> tuple[list[str], np.ndarray]:
        sorted_returns = sorted(self.task_returns.items(), key=lambda kv: kv[0])
        rets = torch.tensor([r[1] for r in sorted_returns])
        # assert (rets <= 1).all().item()
        probs = torch.nn.functional.softmax(-rets / self.temperature, dim=0)
        return [task for task, _ in sorted_returns], probs.numpy()

    def update_returns(self, task_returns: dict[str, float]):
        self.task_returns.update(task_returns)
        if self.stage == 0 and all(v >= self.threshold for v in self.task_returns.values()):
            self.stage = 1
            # new_tasks = {f'!{a} U {b}': 0.0 for a in self.propositions for b in self.propositions if a != b}
            new_tasks = {f'F({a} & (F {b}))': 0.0 for a in self.propositions for b in self.propositions if a != b}
            self.task_returns.update(new_tasks)
        # if self.stage == 1 and (sum(self.task_returns.values()) / len(self.task_returns)) >= self.threshold:
        #     self.stage = 2
        #     new_tasks = {f'!{a} U ({b} & (!{c} U {d}))': 0.0
        #                  for a in self.propositions
        #                  for b in self.propositions
        #                  for c in self.propositions
        #                  for d in self.propositions
        #                  if a != b and c != d and b != d and b != c and a != c and a != d}
        #     self.task_returns.update(new_tasks)