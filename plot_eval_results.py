import pandas as pd
import matplotlib.pyplot as plt

dfs_gcn = [pd.read_csv(f"eval_results/ChessWorld-v1/gcn_formula_big_skip_6_finer/{i}.csv", header=0) for i in range(1, 6)]
dfs_dps = [pd.read_csv(f"eval_results/ChessWorld-v1/deepsets_stay_update_4_finest/{i}.csv", header=0) for i in range(1, 6)]
dfs_tfs = [pd.read_csv(f"eval_results/ChessWorld-v1/transformer_stay/{i}.csv", header=0) for i in range(1, 6)]

mean_df_gcn = pd.concat(dfs_gcn).groupby(level=0).mean()
std_df_gcn = pd.concat(dfs_gcn).groupby(level=0).std()

mean_df_dps = pd.concat(dfs_dps).groupby(level=0).mean()
std_df_dps = pd.concat(dfs_dps).groupby(level=0).std()

mean_df_tfs = pd.concat(dfs_tfs).groupby(level=0).mean()
std_df_tfs = pd.concat(dfs_tfs).groupby(level=0).std()


labels = ["GCN", "Deepsets", "Transformer"]
colors = ["blue", "red", "green"]

mean_dfs = [mean_df_gcn, mean_df_dps, mean_df_tfs]
std_dfs = [std_df_gcn, std_df_dps, std_df_tfs]

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
