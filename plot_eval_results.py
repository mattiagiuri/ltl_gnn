import pandas as pd
import matplotlib.pyplot as plt


def plot_results(env, exp_list, labels, colors, seed_list=None):

    if seed_list is None:
        seed_list = [list(range(1, 6)) for _ in range(len(exp_list))]

    for i, (exp, seeds) in enumerate(zip(exp_list, seed_list)):
        dfs = [pd.read_csv(f"eval_results/{env}/{exp}/{i}.csv", header=0) for i in seeds]

        mean_df = pd.concat(dfs).groupby(level=0).mean()
        std_df = pd.concat(dfs).groupby(level=0).std()

        plt.plot(mean_df.index, mean_df["return"], label=labels[i], color=colors[i])  # Mean line
        plt.fill_between(mean_df.index,
                         mean_df["return"] - std_df["return"],
                         mean_df["return"] + std_df["return"],
                         color=colors[i], alpha=0.2)

    plt.xlabel("Number of Training Steps")
    plt.ylabel("ADR")
    plt.title("ADR over time +- one std")
    plt.legend(title="Models")
    plt.show()


dfs_gcn = [pd.read_csv(f"eval_results/ChessWorld-v1/gcn_formula_big_skip_6_finer/{i}.csv", header=0) for i in range(1, 6)]
dfs_dps = [pd.read_csv(f"eval_results/ChessWorld-v1/deepsets_stay_update_4_finest/{i}.csv", header=0) for i in range(1, 6)]
dfs_tfs = [pd.read_csv(f"eval_results/ChessWorld-v1/transformer_stay/{i}.csv", header=0) for i in range(1, 6)]
dfs_dps_bms = [pd.read_csv(f"eval_results/ChessWorld-v1/deepsets_trial_4/{i}.csv", header=0) for i in range(1, 6)]

mean_df_gcn = pd.concat(dfs_gcn).groupby(level=0).mean()
std_df_gcn = pd.concat(dfs_gcn).groupby(level=0).std()

mean_df_dps = pd.concat(dfs_dps).groupby(level=0).mean()
std_df_dps = pd.concat(dfs_dps).groupby(level=0).std()

mean_df_tfs = pd.concat(dfs_tfs).groupby(level=0).mean()
std_df_tfs = pd.concat(dfs_tfs).groupby(level=0).std()

mean_df_dps_bms = pd.concat(dfs_dps_bms).groupby(level=0).mean()
std_df_dps_bms = pd.concat(dfs_dps_bms).groupby(level=0).std()


labels = ["GCN", "Deepsets", "Transformer", "Deepsets (large avoid)"]
colors = ["blue", "red", "green", "yellow"]

mean_dfs = [mean_df_gcn, mean_df_dps, mean_df_tfs, mean_df_dps_bms]
std_dfs = [std_df_gcn, std_df_dps, std_df_tfs, std_df_dps_bms]

for i, (df_mean, df_std) in enumerate(zip(mean_dfs, std_dfs)):
    plt.plot(df_mean.index, df_mean["return"], label=labels[i], color=colors[i])  # Mean line
    plt.fill_between(df_mean.index,
                     df_mean["return"] - df_std["return"],
                     df_mean["return"] + df_std["return"],
                     color=colors[i], alpha=0.2)  # Confidence interval

# Labels and title
plt.xlabel("Number of Training Steps")
plt.ylabel("ADR")
plt.title("ADR over time +- one std")
plt.legend(title="Models")
plt.show()

#
# plot_results('FlatWorld-v0', ['deepsets_stay', 'gcn_update_2', 'deepsets_update'],
#              ['Deepsets', 'GCN (no pre)', 'Deepsets (prop curriculum)'],
#              ['red', 'blue', 'orange'])

# plot_results('ChessWorld-v1', ['deepsets_update_2', 'gcn_formula_big_skip_6_finer'], ['Deepsets', 'GCN'], ['red', 'blue'])

plot_results('ChessWorld-v1', ['gcn_formula_big_skip_6_finer', 'deepsets_update_2',
                               'deepsets_formula_update', 'gcn_formula_update'],
             ['GCN (cache curriculum)', 'Deepsets (cache curriculum)', 'Deepsets (prop curriculum)', 'GCN (prop curriculum)'],
             ['blue', 'red', 'orange', 'black'])


# plot_results('ChessWorld-v1', ['deepsets_race_update', 'gcn_race_update'], ['Deepsets (2M)', 'GCN (2M)'], ['orange', 'black'])
