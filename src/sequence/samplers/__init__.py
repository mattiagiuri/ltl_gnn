from .curriculum import ZONES_CURRICULUM, LETTER_CURRICULUM, FLATWORLD_CURRICULUM, FLATWORD_PRETRAINING, \
    CHESSWORLD_CURRICULUM, CHESSWORLD_PRETRAINING
from .curriculum_sampler import CurriculumSampler

curricula = {
    'PointLtl2-v0': ZONES_CURRICULUM,
    'LetterEnv-v0': LETTER_CURRICULUM,
    'FlatWorld-v0': FLATWORLD_CURRICULUM,
    'pretraining_FlatWorld-v0': FLATWORD_PRETRAINING,
    'ChessWorld-v0': CHESSWORLD_CURRICULUM,
    'pretraining_ChessWorld-v0': CHESSWORLD_PRETRAINING
}
