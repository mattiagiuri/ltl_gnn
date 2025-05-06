import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Callable, Optional

import numpy as np
import torch

from envs.zones.quadrants import Quadrant
from ltl.automata import LDBASequence
from sequence.samplers.chessworld8_easy_sequence_samplers import chessworld8easy_sample_reach_avoid, \
    chessworld8easy_sample_reach
from sequence.samplers.chessworld8_formula_samplers_update import chessworld_sample_reach_update, \
    chessworld_sample_reach_avoid_update, chessworld_sample_reach_stay_update, chessworld_sample_difficult_ra_update
from sequence.samplers.chessworld_8_sequence_samplers import chessworld8_sample_reach_avoid, chessworld8_sample_reach, \
    chessworld8_sample_reach_stay
from sequence.samplers.chessworld_sequence_samplers import chessworld_sample_reach_avoid, chessworld_sample_reach
from sequence.samplers.flatworld_formula_samplers_update import flatworld_sample_reach_update, \
    flatworld_sample_reach_avoid_update, flatworld_sample_reach_stay_update
from sequence.samplers.flatworld_formula_sequence_samplers import flatworld_sample_simple_reach, \
    flatworld_sample_complex_reach, flatworld_sample_formulae_reach_avoid, flatworld_sample_formula_reach_stay
from sequence.samplers.flatworld_sequence_samplers import flatworld_all_reach_tasks, \
    flatworld_sample_reach_avoid, flatworld_sample_reach_stay, flatworld_sample_reach
from sequence.samplers.sequence_samplers import sample_reach_avoid, all_reach_avoid_tasks, all_reach_tasks, \
    all_reach_stay_tasks, sample_reach_stay

from sequence.samplers.chessworld8_formula_samplers import chessworld8_sample_simple_reach, \
    chessworld8_sample_complex_reach, chessworld8_sample_formulae_reach_avoid, chessworld8_sample_formula_reach_stay
from sequence.samplers.zones_formula_samplers import zonenv_sample_reach_avoid, zonenv_sample_reach


@dataclass
class CurriculumStage(ABC):
    threshold: float | None
    threshold_type: Literal['mean', 'min'] | None

    @abstractmethod
    def sample(self, propositions: list[str]) -> LDBASequence:
        pass

    @abstractmethod
    def update_task_success(self, task_success: dict[LDBASequence, float]) -> None:
        pass


@dataclass
class ExplicitCurriculumStage(CurriculumStage):
    """A curriculum stage in which all tasks are explicitly listed, and sampled from according to previous success."""
    task_fn: Optional[Callable[[list[str]], list[LDBASequence]]]
    eps_task_fn: Optional[Callable[[list[str]], list[LDBASequence]]] = None
    temperature: float = 0.5
    _tasks: list[LDBASequence] | None = None
    _task_success: dict[LDBASequence, float] | None = None

    def sample(self, propositions: list[str]) -> LDBASequence:
        if self._tasks is None:
            self._tasks = []
            if self.task_fn is not None:
                self._tasks += self.task_fn(propositions)
            if self.eps_task_fn is not None:
                self._tasks += self.eps_task_fn(propositions)
        if self._task_success is None:
            return random.choice(self._tasks)
        probs = self.compute_sampling_prob()
        index = np.random.choice(np.arange(len(self._tasks)), p=probs).item()
        return self._tasks[index]

    def compute_sampling_prob(self) -> np.ndarray:
        if len(self._task_success) != len(self._tasks):
            raise ValueError('Task success must be available for all tasks')
        success = torch.tensor([self._task_success[t] for t in self._tasks])
        probs = torch.nn.functional.softmax(-success / self.temperature, dim=0)
        return probs.numpy()

    def update_task_success(self, task_success: dict[LDBASequence, float]) -> None:
        if self._task_success is None:
            self._task_success = {k: v for k, v in task_success.items() if k in self._tasks}
            for t in self._tasks:
                if t not in self._task_success:
                    self._task_success[t] = 0.0
        else:
            self._task_success.update(task_success)


@dataclass
class RandomCurriculumStage(CurriculumStage):
    """A curriculum stage in which tasks are sampled randomly."""
    sampler: Callable[[list[str]], LDBASequence] | Callable[[list[str], dict[str, list[Quadrant]]], LDBASequence]

    def sample(self, propositions: list[str]) -> LDBASequence:
        return self.sampler(propositions)

    def update_task_success(self, task_success: dict[LDBASequence, float]) -> None:
        pass

    def sample_new_zones(self, propositions: list[str], info_dict: dict[str, list[Quadrant]]):
        return self.sampler(propositions, info_dict)

@dataclass
class MultiRandomStage(CurriculumStage):
    """A combination of multiple RandomCurriculumStages with associated sampling probabilities."""
    stages: list[RandomCurriculumStage]
    probs: list[float]

    def sample(self, propositions: list[str]) -> LDBASequence:
        stage = np.random.choice(self.stages, p=self.probs)
        return stage.sample(propositions)

    def update_task_success(self, task_success: dict[LDBASequence, float]) -> None:
        pass

    def sample_new_zones(self, propositions: list[str], info_dict: dict[str, list[Quadrant]]) -> LDBASequence:
        stage = np.random.choice(self.stages, p=self.probs)
        return stage.sample_new_zones(propositions, info_dict)

class Curriculum:
    def __init__(self, stages: list[CurriculumStage]):
        self.stages = stages
        self.stage_index = 0
        self.num_updates = 0

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.stage_index]

    @property
    def finished(self) -> bool:
        return self.stage_index >= len(self.stages)

    def sample(self, propositions: list[str]) -> LDBASequence:
        return self.current_stage.sample(propositions)

    def sample_new_zones(self, propositions: list[str], info_dict: dict[str, list[Quadrant]]) -> LDBASequence:
        return self.current_stage.sample_new_zones(propositions, info_dict)

    def update_task_success(self, task_success: dict[LDBASequence, float], verbose=False) -> None:
        if self.current_stage.threshold is None:
            return
        self.num_updates += 1
        self.num_updates %= 100
        self.current_stage.update_task_success(task_success)
        aggr = np.mean if self.current_stage.threshold_type == 'mean' else np.min
        if aggr(list(task_success.values())) >= self.current_stage.threshold:
            if verbose:
                print('=' * 80)
                print(f"Stage {self.stage_index} completed.")
                print('=' * 80)
            self.stage_index += 1
        else:
            if verbose and self.num_updates % 100 == 0:
                print(f"Stage {self.stage_index} not completed.")
                print(f'MEAN: {np.mean(list(task_success.values()))}, THRESHOLD: {self.current_stage.threshold}')


LETTER_CURRICULUM = Curriculum([
    ExplicitCurriculumStage(
        task_fn=all_reach_avoid_tasks(1),
        temperature=0.1,
        threshold=0.95,
        threshold_type='mean',
    ),
    RandomCurriculumStage(
        sampler=sample_reach_avoid(1, (1, 2), (0, 2)),
        threshold=0.95,
        threshold_type='mean'
    ),
    RandomCurriculumStage(
        sampler=sample_reach_avoid(2, (1, 2), (1, 2)),
        threshold=0.95,
        threshold_type='mean'
    ),
    RandomCurriculumStage(
        sampler=sample_reach_avoid(3, (1, 2), (0, 3)),
        threshold=None,
        threshold_type=None
    ),
])

ZONES_CURRICULUM = Curriculum([
    ExplicitCurriculumStage(  # 0
        task_fn=all_reach_tasks(1),
        temperature=0.5,
        threshold=0.8,
        threshold_type='min',
    ),
    ExplicitCurriculumStage(  # 1
        task_fn=all_reach_tasks(2),
        threshold=0.95,
        threshold_type='mean'
    ),
    ExplicitCurriculumStage(  # 2
        task_fn=all_reach_avoid_tasks(1),
        threshold=0.95,
        threshold_type='mean'
    ),
    ExplicitCurriculumStage(  # 3
        task_fn=all_reach_avoid_tasks(2),
        threshold=0.9,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 4
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid(1, (1, 2), (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_stay(30, (0, 1)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.4, 0.6],
        threshold=0.9,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 5
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid(2, (1, 2), (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_stay(60, (0, 1)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.8, 0.2],
        threshold=0.9,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 6
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid(3, (1, 2), (0, 3)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_stay(60, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.8, 0.2],
        threshold=None,
        threshold_type=None
    ),
])

FLATWORLD_CURRICULUM = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_avoid((1, 2), 1, 1),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.6, 0.4],
        threshold=0.8,
        threshold_type='mean'
    ),
    RandomCurriculumStage(
        sampler=flatworld_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
        threshold=None,
        threshold_type=None
    ),
])

FLATWORD_PRETRAINING = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[1.0, 0.0],
        threshold=None,
        threshold_type=None
    )
    ]
)

CHESSWORLD_CURRICULUM = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld_sample_reach_avoid((1, 2), 1, 1),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld_sample_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.6, 0.4],
        threshold=0.8,
        threshold_type='mean'
    ),
    RandomCurriculumStage(
        sampler=chessworld_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
        threshold=None,
        threshold_type=None
    ),
])

CHESSWORLD_PRETRAINING = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld_sample_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[1.0, 0.0],
        threshold=None,
        threshold_type=None
    )
    ]
)

CHESSWORLD8_CURRICULUM = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8_sample_reach_avoid((1, 2), 1, 1),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.6, 0.4],
        threshold=0.8,
        threshold_type='mean'
    ),
    RandomCurriculumStage(
        sampler=chessworld8_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
        threshold=None,
        threshold_type=None
    ),
])

CHESSWORLD8_PRETRAINING = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[1.0, 0.0],
        threshold=None,
        threshold_type=None
    )
    ]
)


CHESSWORLD8_PRETRAINING_TRANSFORMER = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
                threshold=None,
                threshold_type=None
            ),

            RandomCurriculumStage(
                sampler=chessworld8_sample_reach_stay(5, (0, 2)),
                threshold=None,
                threshold_type=None
            ),

        ],
        probs=[0.8, 0.2],
        threshold=None,
        threshold_type=None
    )
    ]
)

CHESSWORLD8EASY_CURRICULUM = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8easy_sample_reach_avoid((1, 2), 1, 1),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld8easy_sample_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.6, 0.4],
        threshold=0.8,
        threshold_type='mean'
    ),
    RandomCurriculumStage(
        sampler=chessworld8easy_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
        threshold=None,
        threshold_type=None
    ),
])

CHESSWORLD8EASY_PRETRAINING = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8easy_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld8easy_sample_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[1.0, 0.0],
        threshold=None,
        threshold_type=None
    )
    ]
)

CHESSWORLD8_STAY_CURRICULUM = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8_sample_reach_avoid((1, 2), 1, 1),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.6, 0.4],
        threshold=0.8,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_reach_stay(5, (0, 1)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.8, 0.2],
        threshold=0.85,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_reach_stay(10, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.8, 0.2],
        threshold=None,
        threshold_type=None
    ),
    # MultiRandomStage(  # 0
    #     stages=[
    #         RandomCurriculumStage(
    #             sampler=chessworld8_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
    #             threshold=None,
    #             threshold_type=None
    #         ),
    #         RandomCurriculumStage(
    #             sampler=chessworld8_sample_reach_stay(10, (0, 2)),
    #             threshold=None,
    #             threshold_type=None
    #         ),
    #     ],
    #     probs=[0.8, 0.2],
    #     threshold=0.85,
    #     threshold_type='mean'
    # ),
    # MultiRandomStage(  # 0
    #     stages=[
    #         RandomCurriculumStage(
    #             sampler=chessworld8_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
    #             threshold=None,
    #             threshold_type=None
    #         ),
    #         RandomCurriculumStage(
    #             sampler=chessworld8_sample_reach_stay(10, (0, 2)),
    #             threshold=None,
    #             threshold_type=None
    #         ),
    #     ],
    #     probs=[0.8, 0.2],
    #     threshold=0.9,
    #     threshold_type='mean'
    # ),
    # MultiRandomStage(  # 0
    #     stages=[
    #         RandomCurriculumStage(
    #             sampler=chessworld8_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
    #             threshold=None,
    #             threshold_type=None
    #         ),
    #         RandomCurriculumStage(
    #             sampler=chessworld8_sample_reach_stay(10, (0, 2)),
    #             threshold=None,
    #             threshold_type=None
    #         ),
    #     ],
    #     probs=[0.8, 0.2],
    #     threshold=None,
    #     threshold_type=None
    # ),
])

CHESSWORLD8_STAY_PRETRAINING = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_reach_stay(10, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.6, 0.4],
        threshold=None,
        threshold_type=None
    )
    ]
)

CHESSWORLD8_SMALL_STAY = Curriculum([
    RandomCurriculumStage(
        sampler=sample_reach_stay(1, (0, 1)),
        threshold=0.9,
        threshold_type='mean'
    ),
    RandomCurriculumStage(
        sampler=sample_reach_stay(3, (0, 1)),
        threshold=0.9,
        threshold_type='mean'
    ),
    RandomCurriculumStage(
        sampler=sample_reach_stay(5, (0, 1)),
        threshold=0.9,
        threshold_type='mean'
    ),
    RandomCurriculumStage(
        sampler=chessworld8_sample_reach_stay(10, (0, 2)),
        threshold=None,
        threshold_type=None
    )
    ]
)

CHESSWORLD8_STAY_CURRICULUM_PROG = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8_sample_reach_avoid((1, 2), 1, 1),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_stay(5, (0, 1)),
                threshold=0.9,
                threshold_type='mean'
            ),
        ],
        probs=[0.4, 0.3, 0.3],
        threshold=0.9,
        threshold_type='mean'
    ),
    MultiRandomStage(
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
                threshold=0.9,
                threshold_type='mean'
            ),

            RandomCurriculumStage(
                sampler=chessworld8_sample_reach_stay(10, (0, 2)),
                threshold=None,
                threshold_type=None
            )
        ],
        probs=[0.6, 0.4],
        threshold=None,
        threshold_type=None
    )

])

CHESSWORLD8_FORMULA_CURRICULUM = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8_sample_simple_reach((1, 2), (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_complex_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_formulae_reach_avoid((1, 2), 1, (1, 2)),
                threshold=None,
                threshold_type=None
            )
        ],
        probs=[0.2, 0.2, 0.6],
        threshold=0.85,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8_sample_formulae_reach_avoid((1, 2), 1, (1, 2)),
                threshold=None,
                threshold_type=None,

            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_formula_reach_stay(5, (1, 2)),
                threshold=None,
                threshold_type=None
            ),

        ],
        probs=[0.8, 0.2],
        threshold=0.85,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8_sample_formulae_reach_avoid((1, 2), 1, (1, 2)),
                threshold=None,
                threshold_type=None,

            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_formula_reach_stay(10, (1, 2)),
                threshold=None,
                threshold_type=None
            ),

        ],
        probs=[0.8, 0.2],
        threshold=0.85,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8_sample_formulae_reach_avoid((1, 2), 1, (1, 2)),
                threshold=None,
                threshold_type=None,

            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_formula_reach_stay(10, (1, 2)),
                threshold=None,
                threshold_type=None
            ),

        ],
        probs=[0.8, 0.2],
        threshold=0.9,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8_sample_formulae_reach_avoid((1, 2), 1, (1, 2)),
                threshold=None,
                threshold_type=None,

            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_formula_reach_stay(10, (1, 2)),
                threshold=None,
                threshold_type=None
            ),

        ],
        probs=[0.8, 0.2],
        threshold=0.95,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8_sample_formulae_reach_avoid((1, 2), 1, (1, 2)),
                threshold=None,
                threshold_type=None,

            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_formula_reach_stay(10, (1, 2)),
                threshold=None,
                threshold_type=None
            ),

        ],
        probs=[0.8, 0.2],
        threshold=None,
        threshold_type=None
    ),
    ]
)


PRETRAINING_CHESSWORLD8_FORMULA = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8_sample_simple_reach((1, 2), (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_complex_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_formulae_reach_avoid((1, 2), 1, (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_formula_reach_stay(5, (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_formula_reach_stay(10, (1, 2)),
                threshold=None,
                threshold_type=None
            )
        ],
        probs=[0.15, 0.15, 0.4, 0.15, 0.15],
        threshold=None,
        threshold_type=None
    ),
    ]
)

FLATWORLD_STAY_CURRICULUM = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_avoid((1, 2), 1, 1),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.6, 0.4],
        threshold=0.8,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 1
        stages=[
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_stay(5, (0, 1)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.8, 0.2],
        threshold=0.85,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 2
        stages=[
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_stay(10, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.8, 0.2],
        threshold=None,
        threshold_type=None
    ),
    # MultiRandomStage(  # 3
    #     stages=[
    #         RandomCurriculumStage(
    #             sampler=flatworld_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
    #             threshold=None,
    #             threshold_type=None
    #         ),
    #         RandomCurriculumStage(
    #             sampler=flatworld_sample_reach_stay(20, (0, 2)),
    #             threshold=None,
    #             threshold_type=None
    #         ),
    #     ],
    #     probs=[0.8, 0.2],
    #     threshold=None,
    #     threshold_type=None
    # ),
])

FLATWORLD_FORMULA_CURRICULUM = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=flatworld_sample_simple_reach((1, 2), (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_complex_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_formulae_reach_avoid((1, 2), 1, (1, 2)),
                threshold=None,
                threshold_type=None
            )
        ],
        probs=[0.2, 0.2, 0.6],
        threshold=0.85,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=flatworld_sample_formulae_reach_avoid((1, 2), 1, (1, 2)),
                threshold=None,
                threshold_type=None,

            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_formula_reach_stay(5, (1, 2)),
                threshold=None,
                threshold_type=None
            ),

        ],
        probs=[0.8, 0.2],
        threshold=0.85,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=flatworld_sample_formulae_reach_avoid((1, 2), 1, (1, 2)),
                threshold=None,
                threshold_type=None,

            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_formula_reach_stay(10, (1, 2)),
                threshold=None,
                threshold_type=None
            ),

        ],
        probs=[0.8, 0.2],
        threshold=None,
        threshold_type=None
    ),
    # MultiRandomStage(  # 0
    #     stages=[
    #         RandomCurriculumStage(
    #             sampler=flatworld_sample_formulae_reach_avoid((1, 2), 1, (1, 2)),
    #             threshold=None,
    #             threshold_type=None,
    #
    #         ),
    #         RandomCurriculumStage(
    #             sampler=flatworld_sample_formula_reach_stay(20, (1, 2)),
    #             threshold=None,
    #             threshold_type=None
    #         ),
    #
    #     ],
    #     probs=[0.8, 0.2],
    #     threshold=0.85,
    #     threshold_type='mean'
    # ),
    # MultiRandomStage(  # 0
    #     stages=[
    #         RandomCurriculumStage(
    #             sampler=flatworld_sample_formulae_reach_avoid((1, 2), 1, (1, 2)),
    #             threshold=None,
    #             threshold_type=None,
    #
    #         ),
    #         RandomCurriculumStage(
    #             sampler=flatworld_sample_formula_reach_stay(20, (1, 2)),
    #             threshold=None,
    #             threshold_type=None
    #         ),
    #
    #     ],
    #     probs=[0.8, 0.2],
    #     threshold=None,
    #     threshold_type=None
    # ),
    ]
)

PRETRAINING_FLATWORLD_FORMULA = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=flatworld_sample_simple_reach((1, 2), (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_complex_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_formulae_reach_avoid((1, 2), 1, (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_formula_reach_stay(5, (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_formula_reach_stay(10, (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            # RandomCurriculumStage(
            #     sampler=flatworld_sample_formula_reach_stay(20, (1, 2)),
            #     threshold=None,
            #     threshold_type=None
            # )
        ],
        # probs=[0.13, 0.13, 0.35, 0.13, 0.13, 0.13],
        probs=[0.15, 0.15, 0.4, 0.15, 0.15],
        threshold=None,
        threshold_type=None
    ),
    ]
)

PRETRAINING_FLATWORLD_FORMULA_UPDATE = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_update((1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_avoid_update((1, 2), 1, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_stay_update(5, (0, 1)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_stay_update(10, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            # RandomCurriculumStage(
            #     sampler=flatworld_sample_formula_reach_stay(20, (1, 2)),
            #     threshold=None,
            #     threshold_type=None
            # )
        ],
        # probs=[0.13, 0.13, 0.35, 0.13, 0.13, 0.13],
        probs=[0.2, 0.4, 0.2, 0.2],
        threshold=None,
        threshold_type=None
    ),
    ]
)

FLATWORLD_FORMULA_UPDATE = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_update((1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_avoid_update((1, 2), 1, 1),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.4, 0.6],
        threshold=0.8,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_avoid_update((1, 2), 1, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_stay_update(5, (0, 1)),
                threshold=None,
                threshold_type=None
            ),

        ],
        probs=[0.8, 0.2],
        threshold=0.85,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_avoid_update((1, 2), 1, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_stay_update(10, (0, 2)),
                threshold=None,
                threshold_type=None
            ),

        ],
        probs=[0.8, 0.2],
        threshold=None,
        threshold_type=None
    ),
    ]
)


CHESSWORLD8_FORMULA_UPDATE = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld_sample_reach_update((1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld_sample_reach_avoid_update((1, 2), 1, 1),
                threshold=None,
                threshold_type=None
            ),
        RandomCurriculumStage(
                sampler=chessworld_sample_difficult_ra_update(1),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.4, 0.4, 0.2],
        threshold=0.8,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld_sample_reach_avoid_update((1, 2), 1, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld_sample_difficult_ra_update((1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld_sample_reach_stay_update(5, (0, 1)),
                threshold=None,
                threshold_type=None
            ),

        ],
        probs=[0.6, 0.2, 0.2],
        threshold=0.85,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld_sample_reach_avoid_update((1, 2), 1, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld_sample_difficult_ra_update((1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld_sample_reach_stay_update(10, (0, 2)),
                threshold=None,
                threshold_type=None
            ),

        ],
        probs=[0.6, 0.2, 0.2],
        threshold=None,
        threshold_type=None
    ),
    ]
)

PRETRAINING_CHESSWORLD8_FORMULA_UPDATE = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld_sample_reach_update((1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld_sample_reach_avoid_update((1, 2), 1, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
        RandomCurriculumStage(
                sampler=chessworld_sample_difficult_ra_update((1, 2)),
                threshold=None,
                threshold_type=None
            ),
        RandomCurriculumStage(
                sampler=chessworld_sample_reach_stay_update(10, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
        RandomCurriculumStage(
                sampler=chessworld_sample_reach_stay_update(5, (0, 1)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.2, 0.2, 0.2, 0.2, 0.2],
        threshold=None,
        threshold_type=None
    ),
])

RACING_FORMULA_CHESSWORLD8 = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld8_sample_simple_reach((1, 2), (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_complex_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld8_sample_formulae_reach_avoid((1, 2), 1, (1, 2)),
                threshold=None,
                threshold_type=None
            )
        ],
        probs=[0.2, 0.2, 0.6],
        threshold=0.85,
        threshold_type='mean'
    ),

    RandomCurriculumStage(
        sampler=chessworld8_sample_formulae_reach_avoid((1, 2), 1, (1, 2)),
        threshold=None,
        threshold_type=None,

    ),
])

RACING_FORMULA_CHESSWORLD8_UPDATE = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld_sample_reach_update((1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld_sample_reach_avoid_update((1, 2), 1, 1),
                threshold=None,
                threshold_type=None
            ),
        RandomCurriculumStage(
                sampler=chessworld_sample_difficult_ra_update(1),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.4, 0.4, 0.2],
        threshold=0.8,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=chessworld_sample_reach_avoid_update((1, 2), 1, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=chessworld_sample_difficult_ra_update((1, 2)),
                threshold=None,
                threshold_type=None
            ),

        ],
        probs=[0.75, 0.25],
        threshold=None,
        threshold_type=None
    ),
    ]
)


ZONES_UPDATE_CURRICULUM = Curriculum([
    RandomCurriculumStage(
        sampler=zonenv_sample_reach((1, 2)),
        threshold=0.8,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=zonenv_sample_reach_avoid((1, 2), 1, 1),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=zonenv_sample_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.6, 0.4],
        threshold=0.8,
        threshold_type='mean'
    ),
    RandomCurriculumStage(
        sampler=zonenv_sample_reach_avoid((1, 2), 1, (0, 2)),
        threshold=None,
        threshold_type=None
    ),
    ]
)


ZONES_UPDATE_PRETRAINING_CURRICULUM = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=zonenv_sample_reach_avoid((1, 2), 1, 1),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=zonenv_sample_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=zonenv_sample_reach_avoid((1, 2), 1, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.3, 0.3, 0.4],
        # probs=[0.5, 0.5],
        threshold=None,
        threshold_type=None
    ),

    ]
)