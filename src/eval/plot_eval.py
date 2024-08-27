import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(font_scale=.8)


def main():
    env = 'PointLtl2-v0'
    experiments = ['eval']
    name_mapping = {'pre': 'pretraining', 'cat': 'standard'}
    df = process_eval_results(['eval', 'eval2'])
    ax = sns.relplot(df, x='num_steps', y='success_rate_smooth', kind='line', errorbar='sd', hue='experiment')
    ax.set(ylabel='success rate')
    # plt.savefig(os.path.expanduser('~/tmp/plot.png'))
    plt.show()


def process_eval_results(files):
    dfs = []
    for i, file in enumerate(files):
        df = pd.read_csv(f'{file}.csv')
        df['experiment'] = 'try'
        df['seed'] = i
        for col in ['success_rate', 'violation_rate', 'average_steps']:
            df[f'{col}_smooth'] = smooth(df[col], 5)
        dfs.append(df)
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
