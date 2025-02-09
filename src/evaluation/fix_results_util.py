import pandas as pd
import os

path = "chessworld8_ablation/(knight & rook)/"

cur_df = None

for cur_path in os.listdir(path):
    if cur_path.split("/")[-1] == "results.csv":
        continue

    df = pd.read_csv(path + cur_path, index_col=0, header=[0, 1])
    # print(df)

    try:
        df.index = [int(cur_path[-5])]
    except ValueError:
        continue

    # print(df)

    cur_df = pd.concat([df, cur_df], axis=0)

cur_df.sort_index(inplace=True, ascending=True)
cur_df.to_csv(path + "results.csv")
