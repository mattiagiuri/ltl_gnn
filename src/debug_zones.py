import envs
from ltl import EventuallySampler

env = envs.make_env('PointLtl2Debug-v0', EventuallySampler, render_mode='human')

observation = env.reset(seed=32)
print(f'Goal: {observation["goal"]}')

for i in range(5000):
    action = env.action_space.sample()
    observation, reward, terminated, info = env.step(action)

    if terminated:
        print(reward)
        observation = env.reset()
        print(f'Goal: {observation["goal"]}')

env.close()
