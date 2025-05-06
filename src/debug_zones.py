from envs import make_env
from envs.zones.quadrants import Quadrant
from ltl.samplers import AvoidSampler
from sequence.samplers.curriculum_sampler import NewZonesCurriculumSampler
from sequence.samplers.zones_formula_samplers import zonenv_sample_reach
from preprocessing import init_vocab, init_vars, assignment_vocab, var_names
from sequence.samplers.curriculum import RandomCurriculumStage
import time

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

print('Time', end - start)
print(observation)
print(f'Goal: {observation["goal"]}')
print(env.get_zone_quadrants())
print(env.get_agent_quadrant())
print(env.get_propositions())
init_vocab(env.get_possible_assignments())
init_vars(env.get_propositions())

print(assignment_vocab)
print(var_names)

for i in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, info = env.step(action)

    if terminated:
        print(f'Success: {"success" in info}')
        observation = env.reset()
        print(f'Goal: {observation["goal"]}')

env.close()
