import pandas as pd

df = pd.read_csv("data/train.csv", parse_dates=["Date"])

# ---- Day 1 kelvikku answers (text la) ----
print("Avg weekly sales per row, by year:")
print(df.groupby(df["Date"].dt.year)["Weekly_Sales"].mean().round(0))
print("\nTop 3 peak months (avg sales):")
print(df.groupby(df["Date"].dt.month)["Weekly_Sales"].mean().sort_values(ascending=False).head(3).round(0))

# ---- Pothumana data irukkura series mattum edukkurom ----
all_dates = pd.DataFrame({"Date": sorted(df["Date"].unique())})
counts = df.groupby(["Store", "Dept"])["Date"].nunique()
keys = counts[counts >= 90].reset_index()[["Store", "Dept"]]
print("\nSeries kept:", len(keys), "of", len(counts))

# ---- Full weekly calendar, missing weeks = NaN ----
cal = keys.merge(all_dates, how="cross")
panel = cal.merge(df[["Store", "Dept", "Date", "Weekly_Sales"]], on=["Store", "Dept", "Date"], how="left")

# Holiday flag (date level), Store type & size (store level)
panel = panel.merge(df.groupby("Date")["IsHoliday"].first().reset_index(), on="Date", how="left")
panel = panel.merge(df.groupby("Store")[["Type", "Size"]].first().reset_index(), on="Store", how="left")

panel = panel.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
print("\nPanel shape:", panel.shape)
print("Missing Weekly_Sales: %.1f%%" % (panel["Weekly_Sales"].isna().mean() * 100))
panel.to_csv("data/panel.csv", index=False)
print("Saved data/panel.csv")