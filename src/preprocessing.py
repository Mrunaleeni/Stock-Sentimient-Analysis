"""
preprocessing.py
----------------
Text cleaning utilities for the Market Pulse project.

Two explicit pipelines:
  1. clean_for_sentiment()  — DO NOT use before running VADER/FinBERT.
                              VADER and FinBERT expect raw, natural text.
  2. clean_for_analysis()   — Lowercasing + punctuation stripping for
                              word clouds, TF-IDF, and bag-of-words tasks.

The sentiment pipeline in sentiment.py calls clean_for_analysis() only to
populate the `cleaned_headline` column; it never feeds cleaned text to the
NLP models.
"""

import re
import pandas as pd
import os

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_PATH = os.path.join(BASE_DIR, "data", "raw", "news.csv")


# ---------------------------------------------------------------------------
# Core text transforms
# ---------------------------------------------------------------------------

def clean_for_analysis(text: str) -> str:
    """
    Intended for word clouds, TF-IDF, and bag-of-words features.
    - Lowercases
    - Strips punctuation
    - Collapses extra whitespace
    """
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_sentiment(text: str) -> str:
    """
    Light normalisation that is safe to apply before VADER/FinBERT:
    - Strips leading/trailing whitespace
    - Collapses internal whitespace
    Does NOT lowercase or remove punctuation — both hurt VADER/FinBERT accuracy.
    """
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# DataFrame-level helpers
# ---------------------------------------------------------------------------

def preprocess_news(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Loads (or accepts) the news dataframe and adds two derived columns:
      - `raw_headline`     : whitespace-normalised original (safe for NLP models)
      - `cleaned_headline` : lowercased + punctuation-stripped (for TF-IDF etc.)

    Returns the augmented DataFrame.
    """
    if df is None:
        df = pd.read_csv(NEWS_PATH)

    df["raw_headline"]     = df["Headline"].apply(clean_for_sentiment)
    df["cleaned_headline"] = df["Headline"].apply(clean_for_analysis)
    return df


# ---------------------------------------------------------------------------
# TF-IDF helper
# ---------------------------------------------------------------------------

def build_tfidf_matrix(df: pd.DataFrame, max_features: int = 200):
    """
    Builds a TF-IDF matrix from `cleaned_headline`.
    Returns (matrix, feature_names, vectorizer).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=max_features, stop_words="english")
    matrix = vec.fit_transform(df["cleaned_headline"])
    return matrix, vec.get_feature_names_out(), vec


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = preprocess_news()
    print(df[["Headline", "raw_headline", "cleaned_headline"]].head(10).to_string(index=False))

    mat, names, _ = build_tfidf_matrix(df)
    print(f"\nTF-IDF matrix shape: {mat.shape}")
    print("Top terms:", names[:20])
