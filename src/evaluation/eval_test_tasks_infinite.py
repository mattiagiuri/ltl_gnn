import os

import pandas as pd

from evaluation.simulate import simulate


def main():
    exp = 'nodent'
    num_episodes = 500
    tasks = [
        'GF blue & GF green',
        'GF blue & GF green & GF yellow & G !magenta',
        'FG blue',
        'FG blue & F (yellow & F green)'
    ]
    seeds = range(1, 6)
    for task in tasks:
        print(f'Running task: {task}')
        results = []
        for seed in seeds:
            print(f'Running seed: {seed}')
            deterministic = task.startswith('GF')
            accepting_visits = simulate(exp, seed, num_episodes, task, False, False, deterministic=deterministic)
            results.append([seed, accepting_visits])
            df = pd.DataFrame(results, columns=['seed', 'accepting_visits'])
            os.makedirs('multiple_results', exist_ok=True)
            df.to_csv(f'multiple_results/{task}.csv', index=False)


if __name__ == '__main__':
    main()
