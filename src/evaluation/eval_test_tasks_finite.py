import os

import pandas as pd

from evaluation.simulate_zones import simulate


def main():
    exp = 'nodent'
    num_episodes = 500
    tasks = [
        'F (green & (!blue U yellow)) & F magenta',
        'F (blue | green) & F yellow & F magenta',
        '!(magenta | yellow) U (blue & F green)',
        '!green U ((blue | magenta) & (!green U yellow))',
        '((green | blue) => (!yellow U magenta)) U yellow'
    ]
    seeds = range(1, 6)
    for task in tasks:
        print(f'Running task: {task}')
        results = []
        for seed in seeds:
            print(f'Running seed: {seed}')
            sr, mean_steps = simulate(exp, seed, num_episodes, task, True, False, True)
            results.append([seed, sr, mean_steps])
        df = pd.DataFrame(results, columns=['seed', 'success_rate', 'mean_steps'])
        os.makedirs('multiple_results', exist_ok=True)
        df.to_csv(f'multiple_results/{task}.csv', index=False)


if __name__ == '__main__':
    main()
