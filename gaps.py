import pandas as pd

df = pd.read_csv("data/train.csv", parse_dates=["Date"])
counts = df.groupby(["Store", "Dept"])["Date"].nunique()
print("Total weeks in data:", df["Date"].nunique())
print(counts.describe())