from gymnasium.wrappers import FlattenObservation

from envs.ltl2action_wrapper import Ltl2ActionWrapper
from envs.ltl_wrapper import LtlWrapper
from envs.zones.safety_gym_wrapper.safety_gym_wrapper import SafetyGymWrapper
from src.ltl_wrappers import LTLEnv
import safety_gymnasium

env = safety_gymnasium.make('PointLtl2Debug-v0', render_mode="human")
env = SafetyGymWrapper(env)
env = LtlWrapper(env)
env = Ltl2ActionWrapper(env)
env = LTLEnv(env)

observation, info = env.reset(seed=32)

for i in range(5000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation['text'])

    if terminated or truncated:
        observation, info = env.reset()
        # print(f'Goal: {env.label_id_to_color(observation["ltl_state"])}')

env.close()