import pandas as pd
import os
import re

# Get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# File paths
NEWS_PATH = os.path.join(BASE_DIR, "data", "raw", "news.csv")

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    
    return text

def preprocess_news():
    df = pd.read_csv(NEWS_PATH)
    
    # Apply cleaning
    df["Cleaned_Headline"] = df["Headline"].apply(clean_text)
    
    print("Cleaned Data:\n")
    print(df[["Headline", "Cleaned_Headline"]])
    
    return df

if __name__ == "__main__":
    preprocess_news()