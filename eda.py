import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/train.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# 1. Basic checks
print(df.shape)
print(df.head())
print("Date range:", df["Date"].min(), "to", df["Date"].max())
print("Stores:", df["Store"].nunique(), "| Depts:", df["Dept"].nunique())
print("Missing values:\n", df.isna().sum())
print("Negative sales rows:", (df["Weekly_Sales"] < 0).sum())
print(df["Weekly_Sales"].describe())

# 2. Total weekly sales over time
df.groupby("Date")["Weekly_Sales"].sum().plot(title="Total Weekly Sales", figsize=(10, 4))
plt.savefig("weekly_total.png"); plt.close()

# 3. Holiday vs normal week
print("\nAvg sales (holiday vs normal):")
print(df.groupby("IsHoliday")["Weekly_Sales"].mean())

# 4. Month pattern
df["Month"] = df["Date"].dt.month
df.groupby("Month")["Weekly_Sales"].mean().plot(kind="bar", title="Avg Weekly Sales by Month", figsize=(8, 4))
plt.savefig("month_sales.png"); plt.close()

# 5. Store type
print("\nAvg sales by store type:")
print(df.groupby("Type")["Weekly_Sales"].mean())

# 6. Top 10 departments
df.groupby("Dept")["Weekly_Sales"].sum().nlargest(10).plot(kind="barh", title="Top 10 Depts by Sales", figsize=(8, 4))
plt.savefig("top_depts.png"); plt.close()

print("Done. Check the PNG files.")