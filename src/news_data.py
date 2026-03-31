import pandas as pd
import os

# Find project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to news file
NEWS_PATH = os.path.join(BASE_DIR, "data", "raw", "news.csv")

def load_news_data():
    df = pd.read_csv(NEWS_PATH)
    print("News data loaded!\n")
    print(df)
    return df

if __name__ == "__main__":
    load_news_data()