import envs
from envs import make_env
from sequence import RandomSequenceSampler

sampler = RandomSequenceSampler.partial(length=2, unique=True)
env = make_env('PointLtl2Debug-v0', sampler, ltl=False, render_mode='human', max_steps=2000, eval_mode=True)

observation = env.reset(seed=1)
print(f'Goal: {observation["goal"]}')

for i in range(5000):
    action = env.action_space.sample()
    observation, reward, terminated, info = env.step(action)

    if terminated:
        print(reward)
        observation = env.reset()
        print(f'Goal: {observation["goal"]}')

env.close()
