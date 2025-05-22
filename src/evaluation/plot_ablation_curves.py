import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(font_scale=2.9)


def main(update=False):
    # env = 'PointLtl2-v0'
    env = 'ChessWorld-v1'
    # env = 'FlatWorld-v0'
    # experiments = ['noactivesampling', 'nocurriculum']
    experiments = ['GCN (formula update 2)', 'Deepsets (formula update)', 'Transformer (formula update)']
    # experiments = ['deepset_complex', 'gcrl', 'ltl2action']
    name_mapping = {'GCN (formula update)': 'LTL-GNN', 'Deepsets (formula update)': 'DeepLTL', 'GCN (formula update 2)': 'LTL-GNN',
                    'Transformer (formula update)': 'LTL-ENC', 'ltl2action': 'LTL2Action', 'deepset': 'DeepLTL', 'nocurriculum': 'No curriculum', 'deepset_complex': 'DeepLTL'}
    df = process_eval_results(env, experiments, name_mapping, update=update)
    # df_means = df.pivot_table(values='SR', index='pieces', columns='Method', aggfunc='mean').reset_index()

    methods = df['Method'].unique()
    palette = sns.color_palette(n_colors=len(methods))
    method_colors = dict(zip(methods, palette))

    ci = True

    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    ax.set(ylabel='Success Rate', yticks=np.arange(0, 1.01, 0.1), xlabel='Pieces to Avoid', xticks=[1, 2, 3, 4, 5])
    errorbar = ('ci', 90) if ci else ('sd', 1)
    sns.lineplot(df, x='pieces', y='SR', errorbar=errorbar, hue='Method', ax=ax, palette=method_colors)
    mean_df = df.groupby(['pieces', 'Method'], as_index=False)['SR'].mean()

    # Overlay scatter points at the means
    sns.scatterplot(
        mean_df,
        x='pieces',
        y='SR',
        hue='Method',
        ax=ax,
        legend=False,  # don't repeat legend
        s=50,
        marker='o',
        edgecolor='black',
        palette=method_colors
    )
    # sns.relplot(df, x='num_steps', y='return', kind='line', ci=ci, hue='seed', col='Method')
    # plt.savefig(os.path.expanduser('~/work/dphil/iclr-deepltl/figures/training_letter.pdf'))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, fontsize='x-small')  # remove title='Method'

    # for label in ax.xaxis.get_ticklabels()[::2]:
    #     label.set_visible(False)

    plt.savefig('pieces_ablation.pdf', bbox_inches='tight')
    plt.show()


def process_eval_results(env: str, experiments: list[str], name_mapping=None, smooth_radius=5, update=False):
    dfs = []
    for experiment in experiments:
        for seed in range(1, 6):
            path = f'chessworld8_ablation/stay_update/{seed}/results.csv'
            old_df = pd.read_csv(path, index_col=0)
            # print(old_df)

            name = name_mapping.get(experiment, experiment)

            relevant_col = old_df[experiment].to_list()[1:]
            print(relevant_col)

            df = pd.DataFrame({'SR': relevant_col}, index=[1, 2, 3, 4, 5], dtype=float)
            # print(df)
            df['Method'] = name
            df['seed'] = seed
            df['pieces'] = df.index

            if seed == 5 and experiment == 'GCN (formula update)' and update:
                relevant_col = old_df['GCN (quick update)'].to_list()[1:]
                print(relevant_col)

                df = pd.DataFrame({'SR': relevant_col}, index=[1, 2, 3, 4, 5], dtype=float)
                # print(df)
                df['Method'] = name
                df['seed'] = seed
                df['pieces'] = df.index

            dfs.append(df)
            print(df)
        print(f'Loaded 5 files for {name_mapping.get(experiment, experiment)}')
    result = pd.concat(dfs)
    print(result)
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
    main(update=True)
