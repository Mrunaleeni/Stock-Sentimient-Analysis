"""
dashboard/app.py  —  Market Pulse
----------------------------------
Streamlit dashboard for stock sentiment analysis using FinBERT.

Run:
    streamlit run dashboard/app.py

Pre-requisites:
    python src/sentiment.py        # generates data/processed/news_sentiment.csv
    python src/ml_model.py         # generates data/processed/model_results.json
    python src/evaluate_nlp.py     # generates data/processed/confusion_matrices.json
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis import (
    STOCKS,
    SCORE_COL,
    LABEL_COL,
    get_analysis,
    load_sentiment,
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Market Pulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background: #161b22; }
    .stMetric label { font-size: 0.78rem; color: #8b949e; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.4rem; }
    .warning-box {
        background: #2d1b00; border-left: 4px solid #e6a817;
        padding: 10px 14px; border-radius: 6px; margin: 8px 0;
        font-size: 0.88rem;
    }
    .info-box {
        background: #0d1f2d; border-left: 4px solid #1f8ef1;
        padding: 10px 14px; border-radius: 6px; margin: 8px 0;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE      = os.path.join(os.path.dirname(__file__), "..")
SENT_CSV  = os.path.join(BASE, "data", "processed", "news_sentiment.csv")
MODEL_JSON= os.path.join(BASE, "data", "processed", "model_results.json")
CM_JSON   = os.path.join(BASE, "data", "processed", "confusion_matrices.json")


def _sentinel_check():
    missing = [p for p in [SENT_CSV, MODEL_JSON, CM_JSON] if not os.path.exists(p)]
    if missing:
        st.error(
            "**Pre-processing files missing.** Run these scripts first:\n\n"
            "```\npython src/sentiment.py\n"
            "python src/ml_model.py\n"
            "python src/evaluate_nlp.py\n```"
        )
        st.stop()


@st.cache_data(show_spinner="Loading sentiment data…")
def _load_sent():
    return load_sentiment()


@st.cache_data(show_spinner=False)
def _load_json(path):
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _get_analysis(stock):
    return get_analysis(stock)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
_sentinel_check()

st.sidebar.title("⚙️ Controls")
selected_stock = st.sidebar.selectbox("Stock", STOCKS, index=0)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**FinBERT** (ProsusAI/finbert) — BERT transformer fine-tuned on "
    "Financial PhraseBank. Scores are cached in `news_sentiment.csv` "
    "so the model never reruns here."
)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
sent_df = _load_sent()

try:
    merged_df = _get_analysis(selected_stock)
    data_ok   = not merged_df.empty
except Exception as e:
    st.error(f"Could not load analysis data: {e}")
    st.stop()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("📊 Market Pulse")
st.caption(f"Stock sentiment analysis  ·  {selected_stock}  ·  FinBERT")

# ─────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────
latest_price = merged_df["Close"].iloc[-1]   if data_ok else float("nan")
avg_score    = merged_df[SCORE_COL].mean()   if data_ok else float("nan")
pos_pct      = (merged_df[LABEL_COL] == "Positive").mean() * 100 if data_ok else float("nan")
neg_pct      = (merged_df[LABEL_COL] == "Negative").mean() * 100 if data_ok else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Close",        f"₹{latest_price:,.2f}")
c2.metric("Avg Sentiment Score", f"{avg_score:+.3f}")
c3.metric("% Positive Days",     f"{pos_pct:.0f}%")
c4.metric("% Negative Days",     f"{neg_pct:.0f}%")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Price & Sentiment", "📊 Return Analysis",
     "🧪 NLP Evaluation", "📰 Headlines"]
)

# ══════════════════════════════════════════════
# TAB 1 — Price & Sentiment overlay
# ══════════════════════════════════════════════
with tab1:
    st.subheader(f"{selected_stock} — Close Price")

    if data_ok:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=merged_df["Date"], y=merged_df["Close"],
            mode="lines+markers", name="Close Price",
            line=dict(color="#1f8ef1", width=2), marker=dict(size=5),
        ))
        fig.update_layout(
            template="plotly_dark", xaxis_title="Date", yaxis_title="Price (₹)",
            height=380, margin=dict(t=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Daily FinBERT Sentiment Score")
        color_map  = {"Positive": "#2ea043", "Negative": "#da3633", "Neutral": "#8b949e"}
        bar_colors = merged_df[LABEL_COL].map(color_map).fillna("#8b949e")

        fig2 = go.Figure(go.Bar(
            x=merged_df["Date"], y=merged_df[SCORE_COL],
            marker_color=bar_colors, name="Sentiment Score",
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
        fig2.update_layout(
            template="plotly_dark", xaxis_title="Date", yaxis_title="Score",
            height=280, margin=dict(t=20),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No overlapping dates between sentiment and price data for this stock.")

# ══════════════════════════════════════════════
# TAB 2 — Return Analysis
# ══════════════════════════════════════════════
with tab2:
    if data_ok and len(merged_df) >= 3:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Next-Day Return by Sentiment")
            fig3 = px.box(
                merged_df, x=LABEL_COL, y="next_day_return", color=LABEL_COL,
                color_discrete_map={"Positive": "#2ea043", "Negative": "#da3633", "Neutral": "#8b949e"},
                labels={LABEL_COL: "Sentiment", "next_day_return": "Next-Day Return"},
            )
            fig3.update_layout(template="plotly_dark", height=380, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        with col_b:
            st.subheader("Sentiment Distribution")
            counts = merged_df[LABEL_COL].value_counts()
            fig4 = px.pie(
                values=counts.values, names=counts.index, hole=0.5,
                color=counts.index,
                color_discrete_map={"Positive": "#2ea043", "Negative": "#da3633", "Neutral": "#8b949e"},
            )
            fig4.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig4, use_container_width=True)

        st.subheader("FinBERT Score vs Next-Day Return (scatter)")
        fig5 = px.scatter(
            merged_df, x=SCORE_COL, y="next_day_return", color=LABEL_COL,
            color_discrete_map={"Positive": "#2ea043", "Negative": "#da3633", "Neutral": "#8b949e"},
            trendline="ols",
            labels={SCORE_COL: "FinBERT Score", "next_day_return": "Next-Day Return", LABEL_COL: "Sentiment"},
        )
        fig5.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig5, use_container_width=True)

        avg_by_sent = (
            merged_df.groupby(LABEL_COL)["next_day_return"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "Avg Return", "std": "Std Dev", "count": "N"})
            .round(5)
        )
        st.subheader("Average Next-Day Return by Sentiment Label")
        st.dataframe(avg_by_sent, use_container_width=True)
    else:
        st.info("Not enough overlapping data points for return analysis.")

# ══════════════════════════════════════════════
# TAB 3 — NLP Evaluation
# ══════════════════════════════════════════════
with tab3:
    st.subheader("FinBERT Evaluation — Hand-Labeled Headlines")
    st.markdown(
        '<div class="info-box"><strong>Methodology:</strong> 60 headlines were '
        'hand-labeled (Positive / Negative / Neutral) based on clear financial '
        'framing in the text. Accuracy here likely <em>overestimates</em> '
        'real-world performance on noisier, live news data.</div>',
        unsafe_allow_html=True,
    )

    if os.path.exists(CM_JSON):
        cm_data = _load_json(CM_JSON)
        labels  = ["Positive", "Neutral", "Negative"]

        if "finbert" not in cm_data:
            st.info("FinBERT evaluation data not found. Run `python src/evaluate_nlp.py`.")
        else:
            res = cm_data["finbert"]
            acc = res["accuracy"]
            rep = res["report"]
            cm  = res["confusion_matrix"]

            st.metric("Accuracy", f"{acc:.1%}")

            pr_rows = []
            for lbl in labels:
                if lbl in rep:
                    r = rep[lbl]
                    pr_rows.append({
                        "Label":     lbl,
                        "Precision": round(r["precision"], 3),
                        "Recall":    round(r["recall"], 3),
                        "F1":        round(r["f1-score"], 3),
                        "Support":   int(r["support"]),
                    })
            st.dataframe(pd.DataFrame(pr_rows), use_container_width=True, hide_index=True)

            cm_arr  = np.array(cm["matrix"])
            cm_lbls = cm["labels"]
            fig_cm  = px.imshow(
                cm_arr, x=cm_lbls, y=cm_lbls,
                color_continuous_scale="Blues", text_auto=True,
                labels=dict(x="Predicted", y="True", color="Count"),
                title="FinBERT Confusion Matrix",
            )
            fig_cm.update_layout(template="plotly_dark", height=360, margin=dict(t=50))
            st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown(
            '<div class="warning-box">⚠️ <strong>Limitation:</strong> FinBERT was '
            'evaluated on headlines written for this project — relatively unambiguous '
            'and clean. Real financial news can be sarcastic, speculative, or mixed, '
            'and accuracy will likely be lower in production.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Run `python src/evaluate_nlp.py` to generate evaluation data.")

# ══════════════════════════════════════════════
# TAB 4 — Headlines table
# ══════════════════════════════════════════════
with tab4:
    st.subheader(f"Headlines — {selected_stock}")

    stock_headlines = sent_df[sent_df["Stock"] == selected_stock][
        ["Date", "Headline", "fb_label", "fb_score"]
    ].sort_values("Date", ascending=False).reset_index(drop=True)

    if stock_headlines.empty:
        st.info("No headlines found for this stock.")
    else:
        def _row_style(row):
            colors = {
                "Positive": "background-color: #0d2b0d",
                "Negative": "background-color: #2b0d0d",
                "Neutral":  "background-color: #1a1a1a",
            }
            bg = colors.get(row["fb_label"], "")
            return [bg] * len(row)

        st.dataframe(
            stock_headlines.style.apply(_row_style, axis=1),
            use_container_width=True, height=420,
        )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Market Pulse  ·  FinBERT (ProsusAI/finbert)  ·  NSE data via yfinance  ·  "
    "Not investment advice."
)
