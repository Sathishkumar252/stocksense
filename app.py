import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="StockSense", layout="wide")
st.title("StockSense: Demand Forecast & Inventory Planner")

@st.cache_data
def load():
    return pd.read_csv("data/quantiles.csv", parse_dates=["Date"])

q = load()
for c in ["Weekly_Sales", "p10", "p50", "p90"]:
    q[c] = q[c].clip(lower=0)
q["safety_stock"] = q["p90"] - q["p50"]

# ---- Sidebar ----
store = st.sidebar.selectbox("Store", sorted(q["Store"].unique()))
depts = sorted(q[q["Store"] == store]["Dept"].unique())
dept = st.sidebar.selectbox("Department", depts)

s = q[(q["Store"] == store) & (q["Dept"] == dept)].sort_values("Date").set_index("Date")

# ---- Forecast chart ----
st.subheader(f"Store {store}, Dept {dept}: actual vs forecast range")
st.line_chart(s[["Weekly_Sales", "p10", "p50", "p90"]])

c1, c2, c3 = st.columns(3)
c1.metric("Avg forecast (P50)", f"{s['p50'].mean():,.0f}")
c2.metric("Avg safety stock", f"{s['safety_stock'].mean():,.0f}")
c3.metric("Avg order-up-to (P90)", f"{s['p90'].mean():,.0f}")

# ---- Reorder table ----
st.subheader("Weekly reorder recommendations")
tbl = s[["Weekly_Sales", "p50", "safety_stock", "p90"]].round(0)
tbl.columns = ["Actual sales", "Forecast (P50)", "Safety stock", "Order-up-to (P90)"]
st.dataframe(tbl, width="stretch")

# ---- Policy comparison (all series) ----
st.subheader("Policy comparison (all store-dept series)")
actual = q["Weekly_Sales"]
policies = {
    "A: P50 (no buffer)": q["p50"],
    "B: P50 + 10% buffer": q["p50"] * 1.10,
    "C: P90 (quantile)": q["p90"],
}
rows = []
for name, stock in policies.items():
    rows.append({
        "Policy": name,
        "Stockout %": round((actual > stock).mean() * 100, 1),
        "Fill rate %": round(np.minimum(actual, stock).sum() / actual.sum() * 100, 1),
        "Total stock (M units)": round(stock.sum() / 1e6, 2),
    })
st.table(pd.DataFrame(rows))
st.caption("Holdout: Aug to Oct 2012, 4-week-ahead forecasts. Simulation, single period, sales used as demand proxy.")