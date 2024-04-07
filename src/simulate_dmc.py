import os.path

import numpy as np
import torch
import cv2

from envs import make_env
from ltl import EventuallySampler
from model.model import build_model
from model.agent import Agent
from config import model_configs


def render(env) -> tuple[np.ndarray, int]:
    im = env.render()
    cv2.imshow('env', im[..., ::-1])  # RGB to BGR
    key = cv2.waitKey(1)
    return im, key


def main():
    env_name = 'ltl_cartpole'
    exp = 'first'
    save_gif = False

    env = make_env(env_name, EventuallySampler, render_mode='rgb_array')
    config = model_configs['default']
    training_status = torch.load(f'experiments/ppo/{env_name}/{exp}/0/status.pth', map_location='cpu')
    model = build_model(env, training_status, config)
    agent = Agent(model)

    images = []

    obs = env.reset()
    # print(obs['goal'])
    ret = 0
    for i in range(5000):
        im, key = render(env)
        if save_gif:
            images.append(im)
        action = agent.get_action(obs, deterministic=False)
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
    if save_gif:
        # create a gif using imageio
        import imageio
        images = images[::5]  # increase fps by skipping frames
        imageio.mimsave(os.path.expanduser('~/tmp/agent.gif'), images, fps=60, loop=0, subrectangles=True)


if __name__ == '__main__':
    main()
