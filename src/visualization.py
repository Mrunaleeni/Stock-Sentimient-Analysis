import pandas as pd
import matplotlib.pyplot as plt
import os

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCK_PATH = os.path.join(BASE_DIR, "data", "raw", "RELIANCE.csv")

# Load data
df = pd.read_csv(STOCK_PATH)

# Fix columns
if "Date" not in df.columns:
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Date"}, inplace=True)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# Plot
plt.figure()
plt.plot(df["Date"], df["Close"])
plt.title("RELIANCE Stock Price Trend")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()