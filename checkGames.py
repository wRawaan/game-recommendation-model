import pandas as pd

games_df = pd.read_csv("games.csv")
print(games_df.columns.tolist())