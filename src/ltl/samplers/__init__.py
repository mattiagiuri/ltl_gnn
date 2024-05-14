from .ltl_sampler import LTLSampler
from .adaptive_sampler import AdaptiveSampler
from .eventually_sampler import EventuallySampler
from .fixed_sampler import FixedSampler
from .reach_avoid_sampler import ReachAvoidSampler
from .reach_four_sampler import ReachFourSampler
from .reach_two_sampler import ReachTwoSampler

sampler_map = {
    'eventually_sampler': EventuallySampler,
    'reach_avoid': ReachAvoidSampler,
    'adaptive': AdaptiveSampler,
    'reach4': ReachFourSampler,
    'reach2': ReachTwoSampler,
    'partial': FixedSampler.partial_from_formula('F (magenta & F (yellow))'),
    # TODO: try curriculum where we start with eventually and then move to safety. Could also incorporate G!r.
    'until_sampler': FixedSampler.partial_from_formula('!blue U green'),
}

__all__ = ['LTLSampler', 'EventuallySampler', 'FixedSampler', 'ReachFourSampler', 'sampler_map']