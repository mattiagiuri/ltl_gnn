from envs import make_env
from envs.zones.quadrants import Quadrant
from ltl.samplers import AvoidSampler

sampler = AvoidSampler.partial(depth=2, num_conjuncts=1)
env = make_env('PointLtl2Debug-v0', sampler, render_mode='human', max_steps=2000)

observation = env.reset(seed=1)
print(f'Goal: {observation["goal"]}')
print(env.get_zone_quadrants())

for i in range(5000):
    action = env.action_space.sample()
    observation, reward, terminated, info = env.step(action)

    if terminated:
        print(f'Success: {"success" in info}')
        observation = env.reset()
        print(f'Goal: {observation["goal"]}')
        print(env.get_zone_quadrants())
        print(Quadrant.TOP_LEFT in env.get_zone_quadrants()['blue'])

env.close()
