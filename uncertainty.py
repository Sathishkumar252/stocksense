import pandas as pd
import numpy as np
import lightgbm as lgb

p = pd.read_csv("data/panel.csv", parse_dates=["Date"])
p = p.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

# ---- Same features as model.py ----
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

cut = pd.Timestamp("2012-08-03")
train = p[(p["Date"] < cut) & p["Weekly_Sales"].notna()]
test = p[(p["Date"] >= cut) & p["Weekly_Sales"].notna()].copy()

# ---- 3 quantile models: P10, P50, P90 ----
preds = {}
for q in [0.1, 0.5, 0.9]:
    m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=500,
                          learning_rate=0.05, num_leaves=63,
                          random_state=42, verbose=-1)
    m.fit(train[feats], train["Weekly_Sales"], categorical_feature=["Store", "Dept"])
    preds[q] = m.predict(test[feats])
    print("Trained quantile", q)

# Low <= Mid <= High ku sort pannurom
arr = np.sort(np.column_stack([preds[0.1], preds[0.5], preds[0.9]]), axis=1)
test["p10"], test["p50"], test["p90"] = arr[:, 0], arr[:, 1], arr[:, 2]

# ---- Coverage: actual sales P10 to P90 kulla vizhutha? (ideal ~80%) ----
inside = (test["Weekly_Sales"] >= test["p10"]) & (test["Weekly_Sales"] <= test["p90"])
print("\nCoverage (target ~80%%): %.1f%%" % (inside.mean() * 100))
print("Below P10: %.1f%% (ideal ~10%%)" % ((test["Weekly_Sales"] < test["p10"]).mean() * 100))
print("Above P90: %.1f%% (ideal ~10%%)" % ((test["Weekly_Sales"] > test["p90"]).mean() * 100))

wape_p50 = np.abs(test["Weekly_Sales"] - test["p50"]).sum() / np.abs(test["Weekly_Sales"]).sum() * 100
print("Median (P50) WAPE: %.1f%%" % wape_p50)

test[["Store", "Dept", "Date", "Weekly_Sales", "p10", "p50", "p90"]].to_csv("data/quantiles.csv", index=False)
print("Saved data/quantiles.csv")