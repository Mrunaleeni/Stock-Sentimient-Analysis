"""
analysis.py
-----------
Core analysis layer for Market Pulse.

For each of the 5 stocks:
  1. Load per-stock sentiment scores from data/processed/news_sentiment.csv
  2. Average FinBERT scores per date (multiple headlines on one day → one row)
  3. Merge with NEXT-DAY stock return (we know today's sentiment, we
     measure tomorrow's price move)
  4. Compute Pearson correlation + two-tailed p-value
  5. Return merged DataFrame and a correlation summary dict

The correlation between a single day's headline sentiment and the very
next close is expected to be weak — this is stated explicitly in the
results dict and surfaced in the dashboard.

Public API
----------
  get_analysis(stock)        → merged DataFrame for one stock
  get_all_correlations()     → nested dict  stock → {r, p, n, interp}
  load_sentiment()           → full processed CSV as DataFrame
"""

import os
import warnings
import pandas as pd
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT_PATH = os.path.join(BASE_DIR, "data", "processed", "news_sentiment.csv")
STOCK_DIR = os.path.join(BASE_DIR, "data", "raw")

STOCKS      = ["RELIANCE", "TCS", "INFOSYS", "HDFC", "ICICI"]
SCORE_COL   = "fb_score"
LABEL_COL   = "fb_label"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sentiment() -> pd.DataFrame:
    if not os.path.exists(SENT_PATH):
        raise FileNotFoundError(
            f"Sentiment file not found: {SENT_PATH}\n"
            "Run `python src/sentiment.py` first to generate it."
        )
    return pd.read_csv(SENT_PATH, parse_dates=["Date"])


def _load_stock_prices(name: str) -> pd.DataFrame:
    path = os.path.join(STOCK_DIR, f"{name}.csv")
    df   = pd.read_csv(path)

    if "Date" not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)

    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    # Next-day return: today t → tomorrow t+1
    df["next_day_return"] = df["Close"].pct_change().shift(-1)
    return df[["Date", "Close", "next_day_return"]].dropna()


def _interpret(r: float, p: float) -> str:
    """Plain-English interpretation of a Pearson r / p pair."""
    sig = p < 0.05
    strength = (
        "strong" if abs(r) > 0.5
        else "moderate" if abs(r) > 0.3
        else "weak"
    )
    direction = "positive" if r >= 0 else "negative"

    if not sig:
        return (
            f"r = {r:.3f}, p = {p:.3f} — not statistically significant "
            f"(p ≥ 0.05). No reliable linear relationship detected with "
            f"this sample size."
        )
    return (
        f"r = {r:.3f}, p = {p:.3f} — statistically significant "
        f"({strength} {direction} correlation), but r² = {r**2:.3f} "
        f"means sentiment explains only {r**2*100:.1f}% of return variance. "
        f"Treat with caution."
    )


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def get_analysis(stock: str) -> pd.DataFrame:
    """
    Returns a merged DataFrame for one stock using FinBERT scores with columns:
      Date, fb_score, fb_label, Close, next_day_return
    Sentiment is averaged per date before merging.
    """
    sent_df = load_sentiment()

    # Filter to this stock
    stock_sent = sent_df[sent_df["Stock"] == stock][
        ["Date", SCORE_COL, LABEL_COL, "Headline", "cleaned_headline"]
    ].copy()

    # Average score per date; use mode for label (most common sentiment that day)
    def _mode(s):
        m = s.mode()
        return m.iloc[0] if len(m) else np.nan

    daily = (
        stock_sent
        .groupby("Date")
        .agg(
            score      = (SCORE_COL, "mean"),
            label      = (LABEL_COL, _mode),
            n_headlines= (SCORE_COL, "count"),
        )
        .reset_index()
        .rename(columns={"score": SCORE_COL, "label": LABEL_COL})
    )

    prices = _load_stock_prices(stock)
    merged = pd.merge(daily, prices, on="Date", how="inner")
    return merged.sort_values("Date").reset_index(drop=True)


def get_all_correlations() -> dict:
    """
    Returns nested dict:
      { stock: { r, p, n, significant, interpretation } }
    """
    results = {}

    for stock in STOCKS:
        try:
            df = get_analysis(stock)
        except FileNotFoundError:
            results[stock] = {"error": "stock CSV not found"}
            continue

        if len(df) < 3:
            results[stock] = {
                "error": f"Only {len(df)} overlapping dates — too few for correlation"
            }
            continue

        r, p = stats.pearsonr(df[SCORE_COL], df["next_day_return"])
        results[stock] = {
            "r":              round(float(r), 4),
            "p":              round(float(p), 4),
            "n":              int(len(df)),
            "significant":    bool(p < 0.05),
            "interpretation": _interpret(r, p),
        }

    return results


# ---------------------------------------------------------------------------
# Standalone report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Market Pulse — Sentiment × Return Correlation Report")
    print("=" * 60)

    corrs = get_all_correlations()

    for stock, res in corrs.items():
        print(f"\n  {stock}")
        print(f"  {'─'*50}")
        if "error" in res:
            print(f"    [FinBERT] {res['error']}")
        else:
            print(f"    [FinBERT]  n={res['n']}  {res['interpretation']}")
