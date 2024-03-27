import envs
from ltl import EventuallySampler

env = envs.make_env('PointLtl2Debug-v0', EventuallySampler, render_mode='human')

observation, info = env.reset(seed=32)

for i in range(5000):
    action = env.action_space.sample()
    observation, reward, terminated, info = env.step(action)
    print(observation['goal'])

    if terminated:
        print(reward)
        observation, info = env.reset()
        # print(f'Goal: {env.label_id_to_color(observation["ltl_state"])}')

env.close()
