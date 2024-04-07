import torch
import cv2

from envs import make_env
from ltl import EventuallySampler
from model.model import build_model
from model.agent import Agent
from config import model_configs


def render(env) -> int:
    im = env.render()
    im = im[..., ::-1]  # RGB to BGR
    cv2.imshow('env', im)
    key = cv2.waitKey(1)
    return key


def main():
    env_name = 'ltl_cartpole'
    exp = 'first'

    env = make_env(env_name, EventuallySampler, render_mode='rgb_array')
    config = model_configs['default']
    training_status = torch.load(f'experiments/ppo/{env_name}/{exp}/1/status.pth', map_location='cpu')
    model = build_model(env, training_status, config)
    agent = Agent(model)

    obs = env.reset()
    # print(obs['goal'])
    ret = 0
    for i in range(5000):
        key = render(env)
        action = agent.get_action(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ret += reward
        if key == ord('q'):
            break
        elif key == ord('\b'):
            done = True
        if done:
            obs = env.reset()
            print(f'Done! Reward: {ret}')
            # print(obs['goal'])
            ret = 0

    env.close()


if __name__ == '__main__':
    main()
