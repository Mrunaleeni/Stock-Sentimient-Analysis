import yfinance as yf
import pandas as pd
import os

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Correct data path
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")

# 🔥 ADD THIS BACK (IMPORTANT)
stocks = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "HDFC": "HDFCBANK.NS",
    "ICICI": "ICICIBANK.NS"
}

# Date range
start_date = "2018-01-01"
end_date = "2024-01-01"

def download_stock_data():
    for name, ticker in stocks.items():
        print(f"\nDownloading {name} data...")

        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            print(f"Failed to fetch data for {name}")
            continue

        file_path = os.path.join(DATA_PATH, f"{name}.csv")
        data.reset_index().to_csv(file_path, index=False)

        print(f"Saved {name} data to {file_path}")

    print("\nAll stock data downloaded successfully!")

# Run function
if __name__ == "__main__":
    download_stock_data()