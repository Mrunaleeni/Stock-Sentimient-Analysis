# 📊 Market Pulse

### Stock Sentiment Analysis using FinBERT

> Analyzing how financial news sentiment affects Indian stock prices using a domain-fine-tuned transformer model.

---

## 🚀 What It Does

Market Pulse takes financial news headlines, scores them with **FinBERT** (a BERT model trained specifically on financial text), and measures whether that sentiment has any relationship with next-day stock returns for 5 major NSE stocks.

---

## 🏦 Stocks Covered

| Stock | NSE Ticker |
|-------|-----------|
| Reliance Industries | RELIANCE.NS |
| Tata Consultancy Services | TCS.NS |
| Infosys | INFY.NS |
| HDFC Bank | HDFCBANK.NS |
| ICICI Bank | ICICIBANK.NS |

---

## ⚙️ Pipeline

```
News Headlines
      ↓
FinBERT Sentiment Scoring (raw text, no preprocessing)
      ↓
Cached to data/processed/news_sentiment.csv
      ↓
Merge with Next-Day Stock Returns
      ↓
ML Model + NLP Evaluation + Dashboard
```

---

## 📁 Project Structure

```
├── data/
│   ├── raw/
│   │   ├── news.csv                 # 60 financial headlines
│   │   ├── headlines_labeled.csv    # Hand-labeled ground truth
│   │   ├── RELIANCE.csv
│   │   ├── TCS.csv
│   │   ├── INFOSYS.csv
│   │   ├── HDFC.csv
│   │   └── ICICI.csv
│   └── processed/
│       ├── news_sentiment.csv       # FinBERT scores (cached)
│       ├── confusion_matrices.json
│       └── model_results.json
│
├── src/
│   ├── data_collection.py    # yfinance downloader
│   ├── preprocessing.py      # Text cleaning pipelines
│   ├── sentiment.py          # FinBERT inference
│   ├── analysis.py           # Return analysis for all 5 stocks
│   ├── ml_model.py           # LinearRegression on sentiment scores
│   └── evaluate_nlp.py       # Accuracy + confusion matrix
│
├── dashboard/
│   └── app.py                # Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

## 🛠️ How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download stock data**
```bash
python src/data_collection.py
```

**3. Run FinBERT sentiment scoring** *(downloads ~500 MB on first run)*
```bash
python src/sentiment.py
```

**4. Run ML model and evaluation**
```bash
python src/ml_model.py
python src/evaluate_nlp.py
```

**5. Launch the dashboard**
```bash
streamlit run dashboard/app.py
```

---

## 📊 Dashboard

4 tabs — all driven by the cached FinBERT scores:

- **📈 Price & Sentiment** — stock price chart overlaid with daily sentiment bars
- **📊 Return Analysis** — next-day return distribution by sentiment label + scatter
- **🧪 NLP Evaluation** — accuracy, precision/recall/F1, confusion matrix
- **📰 Headlines** — full headline table colour-coded by sentiment

---

## 🤖 Model

**FinBERT** — [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert)

- BERT fine-tuned on the Financial PhraseBank dataset
- Labels: `Positive` / `Negative` / `Neutral`
- Score = P(Positive) − P(Negative) → continuous value in [−1, +1]
- Applied to **raw headlines** — no lowercasing or punctuation stripping (those hurt transformer tokenisation)
- Runs once, cached to CSV — never reruns in the dashboard

---

## 📉 Honest Results

The ML model (LinearRegression on sentiment → next-day return) produces **near-zero cross-validated R²** across all stocks. This is expected and reported honestly — not hidden.

Why the signal is weak:
- Only ~12 overlapping date points per stock after merging
- Next-day returns are driven by macro factors, FII flows, and global markets far more than a single day's headlines
- A production study would need months of real, scraped news data

---

## 🔧 Tech Stack

`Python` · `FinBERT (transformers + torch)` · `pandas` · `scikit-learn` · `Streamlit` · `Plotly` · `yfinance`

---

## ⚠️ Disclaimer

This project is for academic/educational purposes only. Nothing here constitutes financial or investment advice.
