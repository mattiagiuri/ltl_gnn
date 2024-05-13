import torch
from tqdm import trange

from envs import make_env
from ltl import EventuallySampler, ReachFourSampler
from ltl.samplers import ReachAvoidSampler
from ltl.samplers.fixed_sampler import FixedSampler
from ltl.samplers.loop_sampler import LoopSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs

env_name = 'PointLtl2-v0'
exp = 'depth2'
seed = 1

# sampler = FixedSampler.partial_from_formula('F ((green | yellow) & (F (blue | magenta)))')
# sampler = FixedSampler.partial_from_formula('F (green & F (blue & (F magenta & (F yellow))))')
# sampler = FixedSampler.partial_from_formula('F (magenta & F (yellow & F (blue & F green)))')
# sampler = FixedSampler.partial_from_formula('(!magenta U (yellow & (!blue U green)))')
# sampler = ReachFourSampler
# sampler = LoopSampler
sampler = ReachAvoidSampler.partial_from_depth(2)
env = make_env(env_name, sampler, render_mode='human', max_steps=1000)
config = model_configs['default']
training_status = torch.load(f'experiments/ppo/{env_name}/{exp}/{seed}/status.pth', map_location='cpu')
model = build_model(env, training_status, config, None, False)
agent = Agent(model)

num_episodes = 1000

num_successes = 0
num_violations = 0
rets = []

for i in trange(num_episodes):
    obs = env.reset()
    print(obs['goal'])
    done = False
    ret = 0
    while not done:
        action = agent.get_action(obs, deterministic=True)
        action = action.flatten()
        obs, reward, done, info = env.step(action)
        ret += reward
        if done:
            rets.append(ret)
            if ret > 0:
                num_successes += 1
            if ret < 0:
                num_violations += 1
            print(f'Success rate: {num_successes / (i + 1)}')

env.close()
print(f'Success rate: {num_successes / num_episodes}')
print(f'Violation rate: {num_violations / num_episodes}')
print(f'Num total: {num_episodes}')
print('Average return:', sum(rets) / len(rets))
