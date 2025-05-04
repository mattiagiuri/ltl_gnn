from typing import Callable

from ltl.automata import LDBASequence
from sequence.samplers.curriculum import Curriculum
from envs.zones.quadrants import Quadrant


class CurriculumSampler:

    @classmethod
    def partial(cls, curriculum: Curriculum) -> Callable[[list[str]], 'CurriculumSampler']:
        return lambda propositions: cls(curriculum, propositions)

    def __init__(self, curriculum: Curriculum, propositions: list[str]):
        self.curriculum = curriculum
        self.propositions = propositions

    def __call__(self):
        return self.curriculum.sample(self.propositions)


class NewZonesCurriculumSampler:
    @classmethod
    def partial(cls, curriculum) -> Callable[[list[str]], 'NewZonesCurriculumSampler']:
        return lambda propositions: cls(curriculum, propositions)

    def __init__(self, curriculum, propositions: list[str]):
        self.curriculum = curriculum
        self.propositions = propositions

    def __call__(self, info_dict: dict[str, list[Quadrant]]):
        return self.curriculum.sample_new_zones(self.propositions, info_dict)
