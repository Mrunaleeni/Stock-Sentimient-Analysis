import pandas as pd
import os
from nltk.sentiment import SentimentIntensityAnalyzer

# Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_PATH = os.path.join(BASE_DIR, "data", "raw", "news.csv")

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

def analyze_sentiment():
    df = pd.read_csv(NEWS_PATH)
    
    # Apply sentiment
    df["Sentiment"] = df["Headline"].apply(get_sentiment)
    
    print("\nSentiment Analysis Result:\n")
    print(df[["Headline", "Sentiment"]])
    
    return df

if __name__ == "__main__":
    analyze_sentiment()