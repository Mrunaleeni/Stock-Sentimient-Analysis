# 📊 Market Pulse — Stock Sentiment Analysis

> Does news sentiment predict next-day stock returns for Indian large-caps?
> This project runs **FinBERT** on financial headlines, merges the scores
> with NSE price data, evaluates the model against hand labels, and gives
> an honest answer — including when the signal is weak.

---

## Problem Statement

Financial news is often claimed to "move markets", but the relationship
between a single day's headline sentiment and the very next close is hard
to isolate. This project tries to measure it properly:

- Score headlines with **FinBERT** — a BERT transformer fine-tuned on
  financial text — on raw, uncleaned headlines.
- Use **next-day returns** (not same-day) so sentiment precedes the price
  move in time, avoiding look-ahead bias.
- Average sentiment across multiple headlines per date before merging.
- Evaluate the model against 60 hand-labeled headlines with accuracy,
  precision/recall/F1, and a confusion matrix.
- If the predictive signal is weak, say so — honest negative results are
  part of the analysis.

Stocks covered: **Reliance Industries, TCS, Infosys, HDFC Bank, ICICI Bank**
(NSE tickers, 2018–2024 price history via yfinance).

---

## Project Structure

```
├── data/
│   ├── raw/
│   │   ├── news.csv                 # 60 financial headlines with Stock column
│   │   ├── headlines_labeled.csv    # Hand-labeled ground truth (60 rows)
│   │   ├── RELIANCE.csv             # OHLCV from yfinance
│   │   ├── TCS.csv
│   │   ├── INFOSYS.csv
│   │   ├── HDFC.csv
│   │   └── ICICI.csv
│   └── processed/
│       ├── news_sentiment.csv       # FinBERT scores (generated once, cached)
│       ├── confusion_matrices.json  # NLP evaluation results
│       └── model_results.json       # ML regression results
│
├── src/
│   ├── data_collection.py   # Downloads NSE OHLCV via yfinance
│   ├── preprocessing.py     # clean_for_analysis() / clean_for_sentiment()
│   ├── sentiment.py         # FinBERT pipeline → processed CSV
│   ├── analysis.py          # Return analysis for all 5 stocks
│   ├── ml_model.py          # LinearRegression on real FinBERT scores
│   ├── evaluate_nlp.py      # Accuracy + confusion matrix vs hand labels
│   └── visualization.py     # Standalone matplotlib plots
│
├── dashboard/
│   └── app.py               # Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:** `requirements.txt` pins the CPU-only torch wheel. If you
> have a CUDA GPU, replace the `torch` line with the appropriate CUDA
> variant from [pytorch.org](https://pytorch.org/get-started/locally/).

### 2. Download stock price data

```bash
python src/data_collection.py
```

### 3. Run the sentiment pipeline (downloads FinBERT ~500 MB on first run)

```bash
python src/sentiment.py
```

This writes `data/processed/news_sentiment.csv`. The dashboard reads this
cached file — **FinBERT never reruns on every page refresh**.

### 4. Run the ML model and NLP evaluation

```bash
python src/ml_model.py
python src/evaluate_nlp.py
```

### 5. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Your browser opens at `http://localhost:8501`.

---

## Dashboard Features

| Tab | What it shows |
|-----|--------------|
| 📈 Price & Sentiment | Stock price line chart + daily FinBERT sentiment bar chart |
| 📊 Return Analysis | Box plot of next-day returns by sentiment label, scatter with OLS trendline, avg return table |
| 🧪 NLP Evaluation | Accuracy, precision/recall/F1 table, FinBERT confusion matrix heatmap |
| 📰 Headlines | Full headline table colour-coded by sentiment label |

**Sidebar control:** stock dropdown to switch between all 5 stocks.

---

## Sentiment Model

### FinBERT (`ProsusAI/finbert`)
BERT model fine-tuned on the Financial PhraseBank dataset.
Returns class probabilities for Positive / Negative / Neutral.
Signed score = P(Positive) − P(Negative), giving a continuous value in
[−1, +1] analogous to a sentiment compound score.

**Applied to raw, uncleaned headlines** — no lowercasing or punctuation
stripping before inference, as those transformations hurt transformer
tokenisation and degrade accuracy.

Cleaned text (`cleaned_headline` column in the CSV) is kept separately
for word-cloud and TF-IDF tasks only.

---

## Results

> Actual values depend on your data. Run the pipeline to see live numbers.

### NLP Evaluation (60 hand-labeled headlines)

| Model | Expected Accuracy | Notes |
|-------|------------------|-------|
| FinBERT | ~80–85% | Strong on clear positive/negative phrasing; occasionally over-predicts Positive on neutral corporate announcements |

These numbers likely **overestimate** real-world performance — the test
headlines were written for this project and are relatively unambiguous
compared to live financial news.

### ML Model (LinearRegression on fb_score → next-day return)

Cross-validated R² is expected to be near zero or negative. This is an
honest result reflecting two real constraints:

1. **Small sample** — ~12 overlapping date points per stock is far too
   few for a reliable regression. A production study needs months of
   daily news.
2. **Noise** — next-day returns are driven by macro factors, global
   markets, FII flows, and sector rotation. Sentiment is one small
   signal among many.

Results are flagged explicitly in the terminal output when the signal
is too weak to be actionable.

---

## Limitations

- **Dataset size**: 60 headlines across 5 stocks over ~4 weeks is too
  small for statistically reliable results. Scale to thousands of
  headlines spanning years for production-grade analysis.
- **Headline quality**: headlines were written for this project, not
  scraped from a live news feed. Real headlines contain more ambiguity,
  sarcasm, and mixed signals.
- **No fine-tuning**: FinBERT is used off-the-shelf — fine-tuning on
  NSE-specific news would improve domain accuracy.
- **Causality**: correlation ≠ causation. Even a strong association does
  not mean headlines cause returns; both may reflect the same underlying
  events.
- **Not investment advice**: nothing in this project should be used to
  make trading decisions.

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `yfinance` | NSE OHLCV data |
| `transformers` + `torch` | FinBERT inference |
| `pandas`, `numpy`, `scipy` | Data wrangling + statistics |
| `scikit-learn` | LinearRegression, evaluation metrics |
| `streamlit` | Interactive dashboard |
| `plotly` | Charts |
| `matplotlib` | Standalone plots |
| `wordcloud` | Word-cloud visualisation |
| `nltk` | NLTK utilities (tokenisation helpers) |
