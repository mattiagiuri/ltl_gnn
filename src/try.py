import safety_gymnasium

from envs.ltl2action_wrapper import Ltl2ActionWrapper
from envs.ltl_wrapper import LtlWrapper
from envs.zones.safety_gym_wrapper.safety_gym_wrapper import SafetyGymWrapper
from src.ltl_wrappers import LTLEnv
from utils import Agent

env = safety_gymnasium.make('PointLtl2-v0', render_mode="human")
env = SafetyGymWrapper(env)
env = LtlWrapper(env)
env = Ltl2ActionWrapper(env)
env = LTLEnv(env, 'full', None, 0.0)

model_dir = 'storage/emb/train'
agent = Agent(env, env.observation_space, env.action_space, model_dir, ignoreLTL=False, progression_mode='full',
              gnn='RGCN_8x32_ROOT_SHARED', recurrence=1, dumb_ac=False, device='cpu',
              argmax=True, num_envs=1)

obs = env.reset()

for i in range(5000):
    action = agent.get_action(obs)
    obs, reward, done, info = env.step(action)
    print(obs['text'])

    if done:
        obs = env.reset()
        print(obs['text'])

env.close()
