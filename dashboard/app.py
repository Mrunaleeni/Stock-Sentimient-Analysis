import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.analysis import analyze_data
import plotly.graph_objects as go
import plotly.express as px

# -------------------------------
# LOAD DATA
# -------------------------------
merged_df = analyze_data()

# -------------------------------
# PREPARE DATA
# -------------------------------
stock_df = merged_df.copy()

headlines_df = merged_df[["Date", "Headline", "Sentiment"]].copy()

# -------------------------------
# KPI CALCULATIONS
# -------------------------------
latest_price = merged_df["Close"].iloc[-1]

avg_pos = merged_df[merged_df["Sentiment"]=="Positive"]["Return"].mean()
avg_neg = merged_df[merged_df["Sentiment"]=="Negative"]["Return"].mean()
avg_neu = merged_df[merged_df["Sentiment"]=="Neutral"]["Return"].mean()

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Market Pulse", layout="wide")

# -------------------------------
# DARK THEME STYLE
# -------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.block-container {
    padding-top: 2rem;
}
.metric-card {
    background: #161b22;
    padding: 20px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.title("📊 Market Pulse Dashboard")
st.caption("Stock Price vs News Sentiment Analysis")

# -------------------------------
# KPIs
# -------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Latest Price", f"₹{latest_price:.2f}")
col2.metric("Positive Return", f"{avg_pos:.2%}")
col3.metric("Negative Return", f"{avg_neg:.2%}")
col4.metric("Neutral Return", f"{avg_neu:.2%}")

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📈 Price", "📊 Analysis", "📰 Headlines"])

# -------------------------------
# TAB 1 — PRICE
# -------------------------------
with tab1:

    st.subheader("Stock Price Trend")

    fig = px.line(
        stock_df,
        x="Date",
        y="Close",
        title="Stock Price Movement",
        markers=True
    )

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 2 — ANALYSIS
# -------------------------------
with tab2:

    st.subheader("Return vs Sentiment")

    fig2 = px.box(
        merged_df,
        x="Sentiment",
        y="Return",
        color="Sentiment"
    )

    fig2.update_layout(template="plotly_dark")

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sentiment Distribution")

    sentiment_counts = merged_df["Sentiment"].value_counts()

    fig3 = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        hole=0.5
    )

    fig3.update_layout(template="plotly_dark")

    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# TAB 3 — HEADLINES
# -------------------------------
with tab3:

    st.subheader("Latest News Headlines")

    st.dataframe(headlines_df, use_container_width=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Built with Streamlit | Data Science Project")