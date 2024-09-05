import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(font_scale=.8)


def main():
    env = 'PointLtl2-v0'
    experiments = ['eval']
    name_mapping = {'eval': 'DeepLTL', 'gcrl': 'GCRL'}
    df = process_eval_results(env, experiments, name_mapping)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].set(ylabel='Success rate', yticks=np.arange(0, 1.01, 0.1))
    axs[1].set(ylabel='Discounted return', yticks=np.arange(0, 1.01, 0.1))
    sns.lineplot(df, x='num_steps', y='success_rate_smooth', errorbar='sd', hue='Method', ax=axs[0])
    sns.lineplot(df, x='num_steps', y='return_smooth', errorbar='sd', hue='Method', ax=axs[1])
    plt.savefig(os.path.expanduser('~/tmp/plot.png'))
    plt.show()


def process_eval_results(env: str, experiments: list[str], name_mapping=None, smooth_radius=5):
    dfs = []
    for experiment in experiments:
        path = f'eval_results/{env}/{experiment}'
        files = [f for f in os.listdir(path) if f.endswith('.csv')]
        for file in files:
            df = pd.read_csv(f'{path}/{file}')
            name = name_mapping.get(experiment, experiment)
            df['Method'] = name
            df['seed'] = int(file.split('.')[0])
            for col in ['success_rate', 'violation_rate', 'average_steps', 'return']:
                df[f'{col}_smooth'] = smooth(df[col], smooth_radius)
            dfs.append(df)
        print(f'Loaded {len(files)} files for experiment {name_mapping.get(experiment, experiment)}')
    result = pd.concat(dfs)
    if result.isna().any().any():
        print('Warning: data contains NaN values')
    return result


def smooth(row, radius):
    """
    Computes the moving average over the given row of data. Returns an array of the same shape as the original row.
    """
    y = np.ones(radius)
    z = np.ones(len(row))
    return np.convolve(row, y, 'same') / np.convolve(z, y, 'same')


if __name__ == '__main__':
    main()
