import random
from dataclasses import dataclass
from pprint import pprint
from typing import Literal

import numpy as np
import torch

Task = list[tuple[str, str]]
FrozenTask = tuple[tuple[str, str]]


class CurriculumSequenceSampler:  # TODO: rewrite using assignments instead of simply propositions
    def __init__(self, propositions: list[str], stage: int = 1):
        self.propositions = sorted(propositions)
        self.curriculum = [
            CurriculumStage(
                goals=self.reach_goals(1),
                threshold=0.8,
                threshold_type='min'
            ),
            CurriculumStage(
                goals=self.reach_goals(2),
                threshold=0.95,
                threshold_type='mean'
            ),
            CurriculumStage(
                goals=self.reach_avoid_goals(1),
                threshold=0.95,
                threshold_type='mean'
            ),
            CurriculumStage(
                goals=self.reach_avoid_goals(2),
            ),
        ]
        self.stage = stage - 1
        self.goal_success = None
        self.temperature = 0.5
        self.is_adaptive = True

    @property
    def goals(self):
        return self.curriculum[self.stage].goals

    def reach_goals(self, depth: int) -> list[Task]:
        if depth == 0:
            raise ValueError("Depth must be at least 1.")
        if depth == 1:
            return sorted([[(a, 'empty')] for a in self.propositions])
        rec = self.reach_goals(depth - 1)
        result = []
        for task in rec:
            next_goal = task[0][0]
            for a in self.propositions:
                if a != next_goal:
                    result.append([(a, 'empty')] + task)
        return sorted(result)

    def reach_avoid_goals(self, depth: int) -> list[Task]:
        if depth == 0:
            raise ValueError("Depth must be at least 1.")
        if depth == 1:
            return sorted([[(a, b)] for a in self.propositions for b in self.propositions if a != b])
        rec = self.reach_avoid_goals(depth - 1)
        result = []
        for task in rec:
            next_reach, next_avoid = task[0]
            for a in self.propositions:
                for b in self.propositions:
                    if a != b and a != next_avoid and a != next_reach and b != next_avoid and b != next_reach:
                        result.append([(a, b)] + task)
        return sorted(result)

    def sample(self) -> Task:
        if self.goal_success is None:
            return random.choice(self.goals)
        assert len(self.goal_success) == len(self.goals)
        probs = self.compute_sampling_prob()
        index = np.random.choice(np.arange(len(self.goals)), p=probs).item()
        return self.goals[index]

    def compute_sampling_prob(self) -> np.ndarray:
        success = sorted(self.goal_success.items(), key=lambda kv: kv[0])
        success = torch.tensor([r[1] for r in success])
        probs = torch.nn.functional.softmax(-success / self.temperature, dim=0)
        return probs.numpy()

    def update_returns(self, goal_success: dict[FrozenTask, float]):
        if self.goal_success is None:
            self.goal_success = {k: v for k, v in goal_success.items() if list(k) in self.goals}
            for g in self.goals:
                if tuple(g) not in self.goal_success:
                    self.goal_success[tuple(g)] = 0.0
        else:
            self.goal_success.update(goal_success)
        stage = self.curriculum[self.stage]
        if stage.threshold is None:
            return
        aggr = np.mean if stage.threshold_type == 'mean' else np.min
        if aggr(list(self.goal_success.values())) >= stage.threshold:
            print('=' * 80)
            print(f"Stage {self.stage + 1} completed.")
            print('=' * 80)
            self.stage += 1
            self.goal_success = None


@dataclass
class CurriculumStage:
    goals: list[Task]
    threshold: float | None = None
    threshold_type: Literal['mean', 'min'] | None = None


if __name__ == '__main__':
    sampler = CurriculumSequenceSampler(['a', 'b', 'c', 'd'])
    pprint(sampler.goals)
    sampler.update_returns({
        (('a', 'empty'),): 0.9,
        (('b', 'empty'),): 0.84,
        (('c', 'empty'),): 0.8,
        (('d', 'empty'),): 0.82,
    })
    pprint(sampler.goals)
