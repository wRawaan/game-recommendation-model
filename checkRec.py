import pandas as pd

print("Loading recommendations subset...")
# Load just the first 100,000 rows to test locally without memory strain
rec_df = pd.read_csv("recommendations.csv", nrows=100000)

print("\nColumns in recommendations.csv:")
print(rec_df.columns.tolist())

print("\nFirst few rows:")
print(rec_df.head(3))