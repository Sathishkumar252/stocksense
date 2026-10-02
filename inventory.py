import pandas as pd
import numpy as np

q = pd.read_csv("data/quantiles.csv", parse_dates=["Date"])

# Sales negative (returns) na 0 nu eduthukkurom; forecast um 0 ku keela pogakoodathu
q["actual"] = q["Weekly_Sales"].clip(lower=0)
for c in ["p10", "p50", "p90"]:
    q[c] = q[c].clip(lower=0)

# ---- Safety stock + order-up-to level ----
q["safety_stock"] = q["p90"] - q["p50"]
q["order_up_to"] = q["p90"]

# ---- 3 policies ----
policies = {
    "A: P50 (no buffer)": q["p50"],
    "B: P50 + 10% buffer": q["p50"] * 1.10,
    "C: P90 (quantile)": q["p90"],
}

rows = []
for name, stock in policies.items():
    sold = np.minimum(q["actual"], stock)
    rows.append({
        "Policy": name,
        "Stockout %": round((q["actual"] > stock).mean() * 100, 1),
        "Fill rate %": round(sold.sum() / q["actual"].sum() * 100, 1),
        "Total stock units (M)": round(stock.sum() / 1e6, 2),
        "Excess units (M)": round(np.maximum(stock - q["actual"], 0).sum() / 1e6, 2),
    })

res = pd.DataFrame(rows)
print(res.to_string(index=False))

# ---- Oru example series ----
ex = q[(q["Store"] == 1) & (q["Dept"] == 1)].sort_values("Date").tail(4)
print("\nExample (Store 1, Dept 1):")
print(ex[["Date", "actual", "p50", "safety_stock", "order_up_to"]].round(0).to_string(index=False))

q[["Store", "Dept", "Date", "actual", "p50", "safety_stock", "order_up_to"]].to_csv(
    "data/inventory_plan.csv", index=False)
print("\nSaved data/inventory_plan.csv")