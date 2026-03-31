import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_PATH = os.path.join(BASE_DIR, "data", "raw", "news.csv")
STOCK_PATH = os.path.join(BASE_DIR, "data", "raw", "RELIANCE.csv")

# Load
news_df = pd.read_csv(NEWS_PATH)
stock_df = pd.read_csv(STOCK_PATH)

# Fix stock
if "Date" not in stock_df.columns:
    stock_df.reset_index(inplace=True)
    stock_df.rename(columns={"index": "Date"}, inplace=True)

stock_df["Date"] = pd.to_datetime(stock_df["Date"])
stock_df["Close"] = pd.to_numeric(stock_df["Close"], errors="coerce")
stock_df["Return"] = stock_df["Close"].pct_change(fill_method=None)

# Convert sentiment manually
sentiment_map = {
    "Positive": 1,
    "Neutral": 0,
    "Negative": -1
}

# Add sentiment column manually (simple example)
news_df["Sentiment"] = ["Positive", "Neutral", "Negative", "Positive", "Positive"]
news_df["Sentiment_Score"] = news_df["Sentiment"].map(sentiment_map)

news_df["Date"] = pd.to_datetime(news_df["Date"])

# Merge
df = pd.merge(news_df, stock_df, on="Date", how="inner")

# Drop NaN
df = df.dropna()

# Model
X = df[["Sentiment_Score"]]
y = df["Return"]

model = LinearRegression()
model.fit(X, y)

print("\nModel Trained Successfully!")

print("\nPrediction for Positive Sentiment:")
print(model.predict([[1]]))