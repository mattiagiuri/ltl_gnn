import torch
from tqdm import trange

from envs import make_sequence_env
from ltl import EventuallySampler, ReachFourSampler
from ltl.samplers import ReachAvoidSampler
from ltl.samplers.fixed_sampler import FixedSampler
from ltl.samplers.loop_sampler import LoopSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from utils.model_store import ModelStore

env_name = 'PointLtl2-v0'
exp = 'seq'
seed = 2

env = make_sequence_env(env_name, render_mode='human', max_steps=1000)
config = model_configs['default']
model_store = ModelStore(env_name, exp, seed, None)
training_status = model_store.load_training_status(map_location='cpu')
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
        if reward > 0:
            print(reward)
            print(obs['goal'])
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
