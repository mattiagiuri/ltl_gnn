# import random
# from dataclasses import dataclass
# from pprint import pprint
# from typing import Literal, Callable
#
# import numpy as np
# import torch
#
# from ltl.logic import Assignment, FrozenAssignment
# from sequence import RandomSequenceSampler
# from ltl.automata import LDBASequence
#
#
# @dataclass
# class CurriculumStage:
#     sample: Callable[[list[str]], LDBASequence]
#     threshold: float | None = None
#     threshold_type: Literal['mean', 'min'] | None = None
#     random: bool = True  # whether the sequence explicitly enumerates all tasks or randomly samples
#
#
# def sample_reach_avoid(
#         depth: int | tuple[int, int],
#         num_reach: int | tuple[int, int],
#         num_avoid: int | tuple[int, int],
# ) -> Callable[[list[str]], LDBASequence]:
#     def wrapper(propositions: list[str]) -> LDBASequence:
#         def sample_one(last_reach: set[str]):
#             nr = random.randint(*num_reach) if isinstance(num_reach, tuple) else num_reach
#             na = random.randint(*num_avoid) if isinstance(num_avoid, tuple) else num_avoid
#             available = [p for p in propositions if p not in last_reach]
#             reach = random.sample(available, nr)
#             available = [p for p in available if p not in reach]
#             avoid = random.sample(available, na)
#             assert not (set(reach) & set(avoid) or set(reach) & set(last_reach) or set(avoid) & set(last_reach))
#             reach_assignments = frozenset([Assignment.single_proposition(p, propositions).to_frozen() for p in reach])
#             avoid_assignments = frozenset([Assignment.single_proposition(p, propositions).to_frozen() for p in avoid])
#             return reach_assignments, avoid_assignments, reach
#
#         d = random.randint(*depth) if isinstance(depth, tuple) else depth
#         last_reach = set()
#         seq = []
#         for _ in range(d):
#             reach, avoid, reach_props = sample_one(last_reach)
#             seq.append((reach, avoid))
#             last_reach = reach_props
#         return tuple(seq)
#
#     return wrapper
#
#
# LETTER_CURRICULUM = [
#     CurriculumStage(
#         sample=sample_reach_avoid(1, 1, 1),
#         threshold=0.95,
#         threshold_type='mean',
#         random=False
#     ),
#     CurriculumStage(
#         sample=sample_reach_avoid(1, (1, 2), (0, 2)),
#         threshold=0.9,
#         threshold_type='mean'
#     ),
#     CurriculumStage(
#         sample=sample_reach_avoid(2, (1, 2), (1, 2)),
#         threshold=0.85,
#         threshold_type='mean'
#     ),
#     CurriculumStage(
#         sample=sample_reach_avoid(3, (1, 2), (0, 3)),
#     ),
# ]
#
#
# class RandomCurriculumSampler:
#
#     @classmethod
#     def partial(cls) -> Callable[[list[str]], 'RandomCurriculumSampler']:
#         return lambda propositions: cls(propositions)
#
#     def __init__(self, propositions: list[str]):
#         self.propositions = propositions
#         self.curriculum = LETTER_CURRICULUM
#         self.stage_index = 0
#         self.is_adaptive = True  # TODO: implement adaptive sampling
#
#     @property
#     def stage(self):
#         return self.curriculum[self.stage_index]
#
#     def __call__(self):
#         return self.stage.sample(self.propositions)
#
#     def update_goal_success(self, goal_success: dict[LDBASequence, float]):
#         if self.stage.threshold is None or self.stage_index >= len(self.curriculum):
#             return
#         aggr = np.mean if self.stage.threshold_type == 'mean' else np.min
#         if aggr(list(goal_success.values())) >= self.stage.threshold:
#             print('=' * 80)
#             print(f"Stage {self.stage_index + 1} completed.")
#             print('=' * 80)
#             self.stage_index += 1
#
#
# if __name__ == '__main__':
#     sampler = RandomCurriculumSampler(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
#     print(sampler())
#     sampler.update_goal_success({'a': 0.91, 'b': 1.0})
#     print(sampler())
#     sampler.update_goal_success({'a': 0.91, 'b': 1.0})
#     for _ in range(1000):
#         print(sampler())
