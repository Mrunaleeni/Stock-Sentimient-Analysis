"""
ml_model.py
-----------
Trains a simple linear regression to predict next-day stock return from
FinBERT sentiment scores (fb_score).

Inputs:
  - data/processed/news_sentiment.csv  (from sentiment.py)
  - data/raw/<STOCK>.csv               (OHLCV from yfinance)

Design notes
------------
* Uses the REAL fb_score column — no hardcoded values.
* Merges on date after averaging per-day sentiment (multiple headlines
  on the same date are collapsed to one row per stock per date).
* Target variable is the NEXT-DAY closing return (shift -1).
* A simple LinearRegression is fine for an exploratory project.
  R² will likely be very low (close to 0); that is an honest result and
  is reported explicitly rather than hidden.
* Saves results to data/processed/model_results.json.
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT_PATH   = os.path.join(BASE_DIR, "data", "processed", "news_sentiment.csv")
STOCK_DIR   = os.path.join(BASE_DIR, "data", "raw")
RESULT_PATH = os.path.join(BASE_DIR, "data", "processed", "model_results.json")

STOCKS = ["RELIANCE", "TCS", "INFOSYS", "HDFC", "ICICI"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_stock(name: str) -> pd.DataFrame:
    path = os.path.join(STOCK_DIR, f"{name}.csv")
    df   = pd.read_csv(path)

    # yfinance sometimes saves Date as index
    if "Date" not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)

    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Next-day return: today we know the sentiment, tomorrow we see the move
    df = df.sort_values("Date").reset_index(drop=True)
    df["next_day_return"] = df["Close"].pct_change().shift(-1)
    return df[["Date", "Close", "next_day_return"]].dropna()


def build_dataset(stock_name: str, sent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average FinBERT scores per date for one stock, then merge with
    next-day returns.
    """
    stock_sent = sent_df[sent_df["Stock"] == stock_name].copy()

    # Average multiple headlines on the same date
    daily_sent = (
        stock_sent
        .groupby("Date")[["fb_score"]]
        .mean()
        .reset_index()
    )

    stock_df = load_stock(stock_name)
    merged   = pd.merge(daily_sent, stock_df, on="Date", how="inner")
    return merged.dropna(subset=["fb_score", "next_day_return"])


def train_and_evaluate(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Fits LinearRegression, returns a results dict.
    Uses 5-fold CV R² as the primary metric (more honest than train R²).
    """
    X = df[feature_cols].values
    y = df["next_day_return"].values

    if len(df) < 5:
        return {
            "n_samples": len(df),
            "warning": "Too few samples for reliable evaluation",
            "r2_train": None,
            "r2_cv_mean": None,
            "r2_cv_std": None,
            "mae": None,
            "coefficients": {},
        }

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    # 5-fold CV — honest estimate of generalisation
    cv_scores = cross_val_score(model, X, y, cv=min(5, len(df)), scoring="r2")

    return {
        "n_samples":    len(df),
        "r2_train":     round(r2_score(y, y_pred), 4),
        "r2_cv_mean":   round(float(cv_scores.mean()), 4),
        "r2_cv_std":    round(float(cv_scores.std()), 4),
        "mae":          round(mean_absolute_error(y, y_pred), 6),
        "coefficients": {
            col: round(float(coef), 6)
            for col, coef in zip(feature_cols, model.coef_)
        },
        "intercept": round(float(model.intercept_), 6),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ml_pipeline() -> dict:
    if not os.path.exists(SENT_PATH):
        raise FileNotFoundError(
            f"Sentiment file not found: {SENT_PATH}\n"
            "Run `python src/sentiment.py` first."
        )

    sent_df = pd.read_csv(SENT_PATH, parse_dates=["Date"])

    all_results = {}

    for stock in STOCKS:
        print(f"\n{'='*50}")
        print(f"  {stock}")
        print(f"{'='*50}")

        try:
            df = build_dataset(stock, sent_df)
        except FileNotFoundError:
            print(f"  Stock CSV not found — skipping.")
            continue

        if df.empty:
            print(f"  No overlapping dates after merge — skipping.")
            continue

        print(f"  Samples after merge: {len(df)}")

        # --- FinBERT only ---
        fb_res = train_and_evaluate(df, ["fb_score"])

        all_results[stock] = {"finbert": fb_res}

        print(f"\n  [FinBERT]")
        print(f"    Train R²   : {fb_res['r2_train']}")
        print(f"    CV R² mean : {fb_res['r2_cv_mean']}  ±  {fb_res['r2_cv_std']}")
        print(f"    MAE        : {fb_res['mae']}")
        if fb_res["r2_cv_mean"] is not None and fb_res["r2_cv_mean"] < 0.05:
            print("    ⚠  Very weak predictive signal — results not reliable for trading.")

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nModel results saved → {RESULT_PATH}")

    return all_results


if __name__ == "__main__":
    results = run_ml_pipeline()
