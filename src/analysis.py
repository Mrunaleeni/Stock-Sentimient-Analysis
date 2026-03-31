import pandas as pd
import os
from nltk.sentiment import SentimentIntensityAnalyzer

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_PATH = os.path.join(BASE_DIR, "data", "raw", "news.csv")
STOCK_PATH = os.path.join(BASE_DIR, "data", "raw", "RELIANCE.csv")

# Load sentiment analyzer
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)
    compound = score["compound"]
    
    if compound > 0.05:
        return "Positive"
    elif compound < -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_data():
    # Load datasets
    news_df = pd.read_csv(NEWS_PATH)

    # STOCK FIX
    stock_df = pd.read_csv(STOCK_PATH)

    print("Columns BEFORE fix:", stock_df.columns)

    if "Date" not in stock_df.columns:
        stock_df.reset_index(inplace=True)
        stock_df.rename(columns={"index": "Date"}, inplace=True)

    print("Columns AFTER fix:", stock_df.columns)

    # Convert columns
    stock_df["Date"] = pd.to_datetime(stock_df["Date"], errors="coerce")
    print("\nStock Dates Sample:\n", stock_df["Date"].head(10))
    stock_df["Close"] = pd.to_numeric(stock_df["Close"], errors="coerce")
    news_df["Date"] = pd.to_datetime(news_df["Date"], errors="coerce")

    # Sentiment
    news_df["Sentiment"] = news_df["Headline"].apply(get_sentiment)

    # Returns
    stock_df["Return"] = stock_df["Close"].pct_change(fill_method=None)

    # Merge
    merged_df = pd.merge(news_df, stock_df, on="Date", how="inner")

    print("\nMerged Data:\n")
    print(merged_df[["Date", "Headline", "Sentiment", "Return"]])

    # 🔥 ADD THIS HERE (INSIDE FUNCTION)
    print("\nAverage Return by Sentiment:\n")
    print(merged_df.groupby("Sentiment")["Return"].mean())

    return merged_df


if __name__ == "__main__":
    analyze_data()