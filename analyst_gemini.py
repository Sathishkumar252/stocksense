import numpy as np
import pandas as pd
from google import genai
from google.genai import types

client = genai.Client()  # reads GEMINI_API_KEY from the environment
MODEL = "gemini-3.8-flash"  # if you get a 404 or 429, try "gemini-3.8-flash-lite"

q = pd.read_csv("data/quantiles.csv", parse_dates=["Date"])
q = q.rename(columns={"Weekly_Sales": "actual"})
for c in ["actual", "p10", "p50", "p90"]:
    q[c] = q[c].clip(lower=0)


# ---------- Tools (functions the model can call) ----------
def get_forecast(store: int, dept: int) -> dict:
    """Weekly actual sales and P10/P50/P90 forecast for one store and department (holdout Aug to Oct 2012)."""
    print(f"  [tool call] get_forecast(store={store}, dept={dept})")
    s = q[(q["Store"] == int(store)) & (q["Dept"] == int(dept))].sort_values("Date")
    if s.empty:
        return {"error": "No data for this store and dept"}
    s = s[["Date", "actual", "p10", "p50", "p90"]].copy()
    s["Date"] = s["Date"].dt.strftime("%Y-%m-%d")
    return {"rows": s.round(0).to_dict("records")}


def get_top_risk(n: int = 5) -> dict:
    """Top store-department series with the highest stockout risk. stockout_rate_if_p50_only is the share of weeks actual sales exceeded the median forecast. uncertainty is the average (P90 - P50) / P50. Only series with 6+ weeks and an average forecast of 1000+ are included."""
    print(f"  [tool call] get_top_risk(n={n})")
    d = q.copy()
    d["rel_buffer"] = (d["p90"] - d["p50"]) / d["p50"].replace(0, np.nan)
    d["above_p50"] = d["actual"] > d["p50"]
    g = d.groupby(["Store", "Dept"]).agg(
        weeks=("Date", "count"),
        avg_forecast=("p50", "mean"),
        uncertainty=("rel_buffer", "mean"),
        stockout_rate_if_p50_only=("above_p50", "mean"),
    ).reset_index()
    g = g[(g["weeks"] >= 6) & (g["avg_forecast"] >= 1000)]
    g = g.sort_values(["stockout_rate_if_p50_only", "uncertainty"], ascending=False).head(int(n))
    return {"rows": g.round(2).to_dict("records")}


def get_policy_summary() -> dict:
    """Compares 3 inventory policies (P50 only, P50 + 10% buffer, P90) on stockout %, fill rate % and total stock."""
    print("  [tool call] get_policy_summary()")
    pol = {
        "A: P50 only": q["p50"],
        "B: P50 + 10% buffer": q["p50"] * 1.10,
        "C: P90 (quantile)": q["p90"],
    }
    out = []
    for name, stock in pol.items():
        out.append({
            "policy": name,
            "stockout_pct": round(float((q["actual"] > stock).mean() * 100), 1),
            "fill_rate_pct": round(float(np.minimum(q["actual"], stock).sum() / q["actual"].sum() * 100), 1),
            "total_stock_million_units": round(float(stock.sum() / 1e6), 1),
        })
    return {"rows": out}


SYSTEM = (
    "You are StockSense Analyst, an inventory planning assistant for Walmart store-department sales data. "
    "Always use the tools to get numbers and never guess. The data is a holdout of about 13 weeks (Aug to Oct 2012), "
    "so mention the small-sample caveat when relevant. Answer briefly in plain English with the key numbers."
)

chat = client.chats.create(
    model=MODEL,
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM,
        tools=[get_forecast, get_top_risk, get_policy_summary],  # called automatically
    ),
)

if __name__ == "__main__":
    print("StockSense Analyst (Gemini) ready. Ask a question (type 'exit' to quit).")
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in ("exit", "quit"):
            break
        if not text:
            continue
        try:
            resp = chat.send_message(text)
            print("\nAnalyst:", resp.text)
        except Exception as e:
            print("\nError:", e)
