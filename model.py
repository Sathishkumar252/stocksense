import pandas as pd
import numpy as np
import lightgbm as lgb

p = pd.read_csv("data/panel.csv", parse_dates=["Date"])
p = p.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

# ---- Features (4 weeks ahead forecast) ----
H = 4
g = p.groupby(["Store", "Dept"])["Weekly_Sales"]
for lag in [4, 8, 13, 52]:
    p[f"lag_{lag}"] = g.shift(lag)

shifted = g.shift(H)
for w in [4, 13]:
    p[f"roll{w}"] = shifted.groupby([p["Store"], p["Dept"]]).transform(
        lambda s: s.rolling(w, min_periods=1).mean())

p["Week"] = p["Date"].dt.isocalendar().week.astype(int)
p["Month"] = p["Date"].dt.month
p["IsHoliday"] = p["IsHoliday"].astype(int)
p["Type"] = p["Type"].map({"A": 0, "B": 1, "C": 2})

feats = ["Store", "Dept", "Size", "Type", "IsHoliday", "Week", "Month",
         "lag_4", "lag_8", "lag_13", "lag_52", "roll4", "roll13"]

# ---- Time-based split: last 13 weeks = test ----
cut = pd.Timestamp("2012-08-03")
train = p[(p["Date"] < cut) & p["Weekly_Sales"].notna()]
test = p[(p["Date"] >= cut) & p["Weekly_Sales"].notna()].copy()
print("Train rows:", len(train), "| Test rows:", len(test))

# ---- LightGBM ----
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                          num_leaves=63, random_state=42, verbose=-1)
model.fit(train[feats], train["Weekly_Sales"], categorical_feature=["Store", "Dept"])
test["pred"] = model.predict(test[feats])

# ---- Metric: WAPE (lower is better) ----
def wape(a, f):
    return np.abs(a - f).sum() / np.abs(a).sum() * 100

s1 = test[test["lag_4"].notna()]
print("\n[Naive: sales 4 weeks ago]  rows:", len(s1))
print("  Naive WAPE   : %.1f%%" % wape(s1["Weekly_Sales"], s1["lag_4"]))
print("  LightGBM WAPE: %.1f%%" % wape(s1["Weekly_Sales"], s1["pred"]))

s2 = test[test["lag_52"].notna()]
print("\n[Seasonal naive: same week last year]  rows:", len(s2))
print("  Seasonal naive WAPE: %.1f%%" % wape(s2["Weekly_Sales"], s2["lag_52"]))
print("  LightGBM WAPE      : %.1f%%" % wape(s2["Weekly_Sales"], s2["pred"]))

imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
print("\nTop 5 features:\n", imp.head(5))

test[["Store", "Dept", "Date", "Weekly_Sales", "pred"]].to_csv("data/preds.csv", index=False)
print("\nSaved data/preds.csv")