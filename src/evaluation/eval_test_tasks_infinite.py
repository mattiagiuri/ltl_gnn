import os

import pandas as pd

from evaluation.simulate import simulate

import argparse

env_to_tasks = {
    'PointLtl2-v0': [
        'GF blue & GF green',
        'GF blue & GF green & GF yellow & G !magenta',
        'FG blue',
        'FG blue & F (yellow & F green)',
    ],
    'LetterEnv-v0': [
        'GF a & GF b & GF c & GF d & G (!e & !f)',
        'GF (e & (!a U f))',
    ],
    'FlatWorld-v0': [
        'GF (blue & green) & GF (red & magenta)',
        'GF red & GF yellow & GF orange & G !blue',
    ]
}


def main(env, exp):
    num_episodes = 500
    tasks = env_to_tasks[env]
    gamma = {
        'PointLtl2-v0': 0.998,
        'LetterEnv-v0': 0.94,
        'FlatWorld-v0': 0.98
    }[env]
    seeds = range(1, 6)
    results = []
    if os.path.exists(f'results_infinite/{env}.csv'):
        df = pd.read_csv(f'results_infinite/{env}.csv')
        results = df.values.tolist()
    for task in tasks:
        print(f'Running task: {task}')
        for seed in seeds:
            print(f'Running seed: {seed}')
            # deterministic = task.startswith('GF')  # TODO
            deterministic = False
            accepting_visits = simulate(env, gamma, exp, seed, num_episodes, task, False, False, deterministic=deterministic)
            results.append(['DeepLTL', task, seed, accepting_visits])
            df = pd.DataFrame(results, columns=['method', 'task', 'seed', 'accepting_visits'])
            os.makedirs('results_infinite', exist_ok=True)
            df.to_csv(f'results_infinite/{env}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=['PointLtl2-v0', 'LetterEnv-v0', 'FlatWorld-v0'],
                        default='LetterEnv-v0')
    parser.add_argument('--exp', type=str, default='deepltl')
    args = parser.parse_args()
    main(args.env, args.exp)
