"""
sentiment.py
------------
Runs VADER and FinBERT on raw headlines (no preprocessing applied here).
Outputs data/processed/news_sentiment.csv with columns:
  Date, Headline, Stock, vader_score, vader_label,
  fb_score, fb_label, cleaned_headline

Run this once to generate the processed file.  The dashboard reads the
cached CSV so FinBERT never reruns on every page refresh.
"""

import os
import re
import pandas as pd
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_PATH   = os.path.join(BASE_DIR, "data", "raw",       "news.csv")
OUT_PATH    = os.path.join(BASE_DIR, "data", "processed", "news_sentiment.csv")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_vader():
    try:
        SentimentIntensityAnalyzer().polarity_scores("test")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def label_from_compound(score: float) -> str:
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    return "Neutral"


def clean_for_nlp(text: str) -> str:
    """Lowercase + strip punctuation — used for word clouds / TF-IDF only."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


# ---------------------------------------------------------------------------
# VADER scoring  (operates on raw, uncleaned text)
# ---------------------------------------------------------------------------
def score_vader(headlines: pd.Series) -> pd.DataFrame:
    ensure_vader()
    sia = SentimentIntensityAnalyzer()
    scores = headlines.apply(lambda h: sia.polarity_scores(h)["compound"])
    labels = scores.apply(label_from_compound)
    return pd.DataFrame({"vader_score": scores, "vader_label": labels})


# ---------------------------------------------------------------------------
# FinBERT scoring  (operates on raw, uncleaned text)
# ---------------------------------------------------------------------------
def score_finbert(headlines: pd.Series) -> pd.DataFrame:
    """
    Uses ProsusAI/finbert (finance-domain BERT fine-tuned on financial PhraseBank).
    Returns a signed continuous score: +1 * positive_prob - 1 * negative_prob,
    analogous to VADER's compound, and the argmax label.
    """
    print("Loading FinBERT model (downloads on first run ~500 MB)…")
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # finbert label order: positive=0, negative=1, neutral=2
    id2label = {v: k for k, v in model.config.label2id.items()}

    fb_scores, fb_labels = [], []

    with torch.no_grad():
        for text in headlines:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            logits = model(**enc).logits
            probs  = torch.softmax(logits, dim=-1).squeeze()

            # Map to label names regardless of ordering in this checkpoint
            label_probs = {id2label[i].lower(): probs[i].item() for i in range(len(probs))}

            # Signed score: positive − negative
            signed = label_probs.get("positive", 0.0) - label_probs.get("negative", 0.0)

            # Argmax label, capitalised to match VADER convention
            raw_label = max(label_probs, key=label_probs.get)
            fb_scores.append(round(signed, 4))
            fb_labels.append(raw_label.capitalize())

    return pd.DataFrame({"fb_score": fb_scores, "fb_label": fb_labels})


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_sentiment_pipeline(force: bool = False) -> pd.DataFrame:
    """
    Reads news.csv, scores with VADER + FinBERT, writes news_sentiment.csv.
    Pass force=True to re-run even if the output already exists.
    """
    if not force and os.path.exists(OUT_PATH):
        print(f"Cached sentiment file found at {OUT_PATH}. Loading…")
        return pd.read_csv(OUT_PATH, parse_dates=["Date"])

    df = pd.read_csv(NEWS_PATH, parse_dates=["Date"])
    print(f"Loaded {len(df)} headlines from {NEWS_PATH}")

    # --- VADER (raw headlines) ---
    print("Running VADER…")
    vader_df = score_vader(df["Headline"])

    # --- FinBERT (raw headlines) ---
    finbert_df = score_finbert(df["Headline"])

    # --- Cleaned text kept for word clouds / TF-IDF only ---
    df["cleaned_headline"] = df["Headline"].apply(clean_for_nlp)

    # --- Combine ---
    result = pd.concat([df, vader_df, finbert_df], axis=1)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"Saved sentiment scores → {OUT_PATH}")

    return result


if __name__ == "__main__":
    df = run_sentiment_pipeline(force=True)
    print(df[["Date", "Stock", "Headline", "vader_label", "fb_label"]].to_string(index=False))
