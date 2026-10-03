# 📊 Market Pulse — Stock Sentiment Analysis

> Can the way financial news is written predict how a stock moves the next day?

---

## The Idea

Every day, hundreds of financial headlines are published about listed companies — earnings beats, regulatory probes, deal wins, leadership changes. Some of this news is clearly good. Some is clearly bad. But does the *sentiment* of that language actually show up in stock prices?

Market Pulse tries to answer that question for 5 major Indian stocks — **Reliance Industries, TCS, Infosys, HDFC Bank, and ICICI Bank** — by running a state-of-the-art NLP model on financial headlines and measuring whether the sentiment score has any relationship with how the stock moves the *following* day.

---

## Why FinBERT

Most sentiment tools were built for social media — Twitter posts, product reviews, Reddit comments. Financial language is different. A headline like *"HDFC Bank gross NPA ratio rises to 1.4%"* is clearly bad news, but a general-purpose model reads "rises" and scores it as positive.

**FinBERT** is a BERT transformer that was fine-tuned specifically on the Financial PhraseBank dataset — thousands of sentences from financial news, hand-labeled by finance professionals. It understands domain-specific phrasing that generic models miss.

One important design choice: FinBERT is run on **raw, uncleaned headlines**. No lowercasing. No punctuation stripping. Transformers use subword tokenisation and are sensitive to capitalisation and punctuation — cleaning the text before inference would actually hurt the model's accuracy.

A separate cleaned version of each headline is kept in the dataset, but only for auxiliary tasks like word clouds and TF-IDF.

---

## What the Project Actually Does

**Sentiment Scoring**
Each headline is passed through FinBERT, which outputs probabilities for Positive, Negative, and Neutral. A signed score is computed as P(Positive) − P(Negative), giving a continuous value between −1 and +1. The results are saved to a CSV once and never recomputed — so the 500 MB model doesn't run every time the dashboard loads.

**Return Analysis**
The sentiment scores are merged with historical stock price data. The key design choice here is using **next-day returns** — the stock's percentage change on the day *after* the headline, not the same day. This avoids look-ahead bias: in a real scenario, you read the news today and observe the market reaction tomorrow.

When multiple headlines exist for the same stock on the same date, their scores are averaged into a single daily sentiment value before merging.

**NLP Evaluation**
60 headlines were hand-labeled as Positive, Negative, or Neutral based on their financial framing. FinBERT's predictions are compared against these labels to produce accuracy, precision, recall, F1-score, and a confusion matrix. This is proper NLP evaluation methodology — not just eyeballing outputs.

**ML Model**
A simple linear regression is trained to predict next-day return from the FinBERT score. The results are reported with 5-fold cross-validated R² and MAE. The model is honest about weak results — near-zero R² is flagged explicitly rather than swept under the rug.

**Dashboard**
An interactive Streamlit dashboard lets you switch between all 5 stocks and explore the data across four views: price and sentiment overlay, return distribution by sentiment label, the NLP evaluation results, and a colour-coded headline table.

---

## What the Results Show

The honest answer: **the signal is weak**.

Cross-validated R² is near zero for all five stocks. This isn't a failure of the pipeline — it's an accurate reflection of how hard short-term return prediction is. Next-day stock moves are driven by macro conditions, FII flows, global markets, and sector rotation. A handful of headlines per stock per day is a tiny signal in a very noisy system.

FinBERT itself performs well on the hand-labeled set (~80–85% accuracy), which confirms the NLP part is working correctly. The weakness is in the financial signal, not the model.

This is worth stating clearly: **a clean pipeline with honest negative results is more valuable than a misleading one with inflated metrics.**

---

## Stocks Covered

| Company | Exchange |
|---------|---------|
| Reliance Industries | NSE |
| Tata Consultancy Services | NSE |
| Infosys | NSE |
| HDFC Bank | NSE |
| ICICI Bank | NSE |

Price data covers **2018–2024** sourced via yfinance.

---

## Tech Stack

`Python` · `FinBERT (ProsusAI/finbert)` · `HuggingFace Transformers` · `PyTorch` · `pandas` · `scikit-learn` · `scipy` · `Streamlit` · `Plotly` · `yfinance`

---

## Limitations

- **60 headlines is a small dataset.** Reliable statistical conclusions need thousands of data points spanning months or years of real news.
- **Headlines were written for this project**, not scraped from a live news feed. Real headlines are messier, more ambiguous, and occasionally sarcastic.
- **FinBERT is used off-the-shelf.** Fine-tuning it on NSE-specific news would push accuracy higher.
- **Correlation is not causation.** Even where a relationship exists, headlines and returns may both be reacting to the same underlying event rather than one causing the other.

---

*Built as an NLP subject project. Not financial advice.*
