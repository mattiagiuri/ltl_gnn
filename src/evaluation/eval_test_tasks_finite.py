import argparse
import os

import pandas as pd

from evaluation.simulate import simulate


env_to_tasks = {
    'PointLtl2-v0': [
        'F (green & (!blue U yellow)) & F magenta',
        'F (blue | green) & F yellow & F magenta',
        '!(magenta | yellow) U (blue & F green)',
        '!green U ((blue | magenta) & (!green U yellow))',
        '((green | blue) => (!yellow U magenta)) U yellow'
    ],
    'LetterEnv-v0': [
        'F (a & (!b U c)) & F d',
        '(F d) & (!f U (d & F b))',
        '(F ((a | c | j) & F b)) & (F (c & F d)) & F k',
        '!a U (b & (!c U (d & (!e U f))))',
        '((a | b | c | d) => F (e & (F (f & F g)))) U (h & F i)'
    ]
}


def main(env, exp, deterministic):
    num_episodes = 500
    tasks = env_to_tasks[env]
    gamma = 0.94 if env == 'LetterEnv-v0' else 0.998
    seeds = range(1, 6)
    results = []
    if os.path.exists(f'results/{env}.csv'):
        df = pd.read_csv(f'results/{env}.csv')
        results = df.values.tolist()
    for task in tasks:
        print(f'Running task: {task}')
        for seed in seeds:
            print(f'Running seed: {seed}')
            sr, mean_steps = simulate(env, gamma, exp, seed, num_episodes, task, True, False, deterministic)
            results.append(['DeepLTL', task, seed, sr, mean_steps])
            df = pd.DataFrame(results, columns=['method', 'task', 'seed', 'success_rate', 'mean_steps'])
            os.makedirs('results', exist_ok=True)
            df.to_csv(f'results/{env}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=['PointLtl2-v0', 'LetterEnv-v0'], default='LetterEnv-v0')
    parser.add_argument('--exp', type=str, default='final')
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    main(args.env, args.exp, args.deterministic)
