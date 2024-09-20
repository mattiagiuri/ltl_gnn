from .curriculum import ZONES_CURRICULUM, LETTER_CURRICULUM, FLATWORLD_CURRICULUM
from .curriculum_sampler import CurriculumSampler

curricula = {
    'PointLtl2-v0': ZONES_CURRICULUM,
    'LetterEnv-v0': LETTER_CURRICULUM,
    'FlatWorld-v0': FLATWORLD_CURRICULUM
}
