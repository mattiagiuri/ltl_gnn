from envs import make_env
from envs.zones.quadrants import Quadrant
from ltl.samplers import AvoidSampler
from sequence.samplers.curriculum_sampler import NewZonesCurriculumSampler
from sequence.samplers.zones_formula_samplers import zonenv_sample_reach
from preprocessing import init_vocab, init_vars, assignment_vocab, var_names
from sequence.samplers.curriculum import RandomCurriculumStage
import time
from model.formulae_utils.ContextMaker import ContextMaker

sampler_wrapper = RandomCurriculumStage(sampler=zonenv_sample_reach(2), threshold_type=None, threshold=None)
sampler = NewZonesCurriculumSampler.partial(sampler_wrapper)
# sampler = AvoidSampler.partial(depth=2, num_conjuncts=1)
env = make_env('PointLtl2Debug-v0', sampler, render_mode='human', max_steps=2000, areas_mode=True, sequence=True)
# print(env.areas_mode)
# print(env.observation_space.spaces.keys())
# env = make_env('PointLtl2-v0', sampler, render_mode='human', max_steps=2000)

start = time.time()
observation = env.reset(seed=1)
end = time.time()

true_vars = env.get_propositions()
print(true_vars)
var_names = list(true_vars) + ['EPSILON', 'NULL', 'blank']
augment_neg = ['!right', '!top']

sample_voc = {0: 'PAD', 1: 'EPSILON', 2: 'NULL', 3: 'blue', 4: 'green', 5: 'magenta', 6: 'yellow', 7: 'right', 8: 'top',
              9: 'right&blue', 10: 'right&green', 11: 'right&magenta', 12: 'right&yellow', 13: 'top&blue',
              14: 'top&green', 15: 'top&magenta', 16: 'top&yellow', 17: 'right&top', 18: 'right&top&blue',
              19: 'right&top&green', 20: 'right&top&magenta', 21: 'right&top&yellow', 22: 'blank'}

cm = ContextMaker(sample_voc, var_names, true_vars, augment_neg)
cm.generate_cache()
cm.check_cache_correctness()

print(len(cm.cache))
# for k, v in cm.cache.items():
#     print(k, v)


