import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
import requests
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
#  ADVANCED STOCK ANALYSIS TOOL (Company name + Symbol Search)
# ============================================================

def find_ticker_from_name(query):
    """
    Search for a stock ticker symbol by company name using Yahoo Finance API.
    """
    try:
        print(f"\n🔍 Searching for company: {query} ...")
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        data = response.json()

        if "quotes" in data and len(data["quotes"]) > 0:
            first = data["quotes"][0]
            symbol = first.get("symbol", "N/A")
            name = first.get("shortname", query)
            print(f"✅ Found: {name} ({symbol})")
            return symbol
        else:
            print("⚠️ No matching ticker found.")
            return None
    except Exception as e:
        print(f"⚠️ Error searching for company: {e}")
        return None


def get_stock_data(ticker, period="6mo", interval="1d"):
    """
    Fetch stock data safely from Yahoo Finance.
    """
    try:
        print(f"\n📡 Fetching data for {ticker} ...")
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period=period, interval=interval)

        if data.empty:
            raise ValueError("Empty dataset received. The symbol might be delisted or invalid.")
        data.reset_index(inplace=True)
        return data, ticker_obj

    except Exception as e:
        print(f"⚠️ Error fetching data for {ticker}: {e}")
        return None, None


def show_company_info(ticker_obj, ticker):
    """
    Displays basic company information (if available).
    """
    try:
        info = ticker_obj.info
        print("\n🏢 Company Information")
        print("=" * 60)
        print(f"Name: {info.get('longName', 'N/A')}")
        print(f"Symbol: {ticker}")
        print(f"Sector: {info.get('sector', 'N/A')}")
        print(f"Industry: {info.get('industry', 'N/A')}")
        print(f"Market Cap: {info.get('marketCap', 'N/A'):,}")
        print(f"Country: {info.get('country', 'N/A')}")
        print(f"Website: {info.get('website', 'N/A')}")
    except Exception:
        print("⚠️ Company information not available.")


def show_basic_stats(df):
    """
    Displays basic statistical summary of the stock prices.
    """
    print("\n📈 Basic Price Statistics")
    print("=" * 60)
    stats = df.describe()[["Open", "High", "Low", "Close", "Volume"]]
    print(stats.round(2))
    print("=" * 60)


def calculate_indicators(df):
    """
    Adds useful indicators: moving averages, returns, volatility.
    """
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["Daily Return"] = df["Close"].pct_change()
    df["Volatility (20D)"] = df["Daily Return"].rolling(window=20).std() * np.sqrt(20)
    return df


def plot_stock_data(df, ticker):
    """
    Generates and displays key stock analysis charts.
    """
    print("\n📊 Generating visual charts...")

    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Close"], label="Close Price", linewidth=1.8)
    plt.plot(df["Date"], df["MA20"], label="20-Day MA", linestyle="--")
    plt.plot(df["Date"], df["MA50"], label="50-Day MA", linestyle="--")
    plt.title(f"{ticker} Closing Price and Moving Averages", fontsize=13)
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.bar(df["Date"], df["Volume"], color="skyblue")
    plt.title(f"{ticker} Trading Volume", fontsize=13)
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.tight_layout()
    plt.show()


def show_volatility(df, ticker):
    """
    Displays recent volatility and risk insights.
    """
    recent_vol = df["Volatility (20D)"].iloc[-1]
    avg_vol = df["Volatility (20D)"].mean()
    print("\n⚡ Volatility Analysis")
    print("=" * 60)
    print(f"Current 20-Day Volatility: {recent_vol:.4f}")
    print(f"Average Volatility (20-Day Rolling): {avg_vol:.4f}")
    print(f"Max Volatility in period: {df['Volatility (20D)'].max():.4f}")
    print("=" * 60)


def predict_next_close(df):
    """
    Predicts the next closing price using Linear Regression on historical data.
    """
    df = df.dropna(subset=["Close"]).copy()
    df["Day"] = np.arange(len(df))
    X = df[["Day"]]
    y = df["Close"]
    model = LinearRegression()
    model.fit(X, y)
    next_day = np.array([[len(df)]])
    predicted = model.predict(next_day)[0]
    slope = model.coef_[0]

    print("\n📉 Price Prediction Summary")
    print("=" * 60)
    print(f"Trend: {'📈 Upward' if slope > 0 else '📉 Downward'}")
    print(f"Predicted Next Close: ${predicted:.2f}")
    print("=" * 60)
    return predicted


def plot_regression_trend(df, ticker):
    """
    Visualizes regression trend line over price data.
    """
    df = df.dropna(subset=["Close"]).copy()
    df["Day"] = np.arange(len(df))
    model = LinearRegression()
    model.fit(df[["Day"]], df["Close"])
    predictions = model.predict(df[["Day"]])

    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Close"], label="Actual Close", linewidth=2)
    plt.plot(df["Date"], predictions, label="Trend Line", linestyle="--", color="red")
    plt.title(f"{ticker} Price Trend (Linear Regression)")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_daily_return_stats(df):
    """
    Displays return metrics and risk.
    """
    print("\n💹 Daily Return Metrics")
    print("=" * 60)
    mean_return = df["Daily Return"].mean()
    std_return = df["Daily Return"].std()
    sharpe_ratio = mean_return / std_return * np.sqrt(252)
    print(f"Avg Daily Return: {mean_return:.5f}")
    print(f"Std Deviation (Risk): {std_return:.5f}")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print("=" * 60)


def export_to_csv(df, ticker):
    """
    Saves stock data with indicators to a CSV file.
    """
    filename = f"{ticker}_analysis.csv"
    df.to_csv(filename, index=False)
    print(f"\n💾 Data exported successfully to {filename}")


# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("🧠 ADVANCED STOCK ANALYSIS TOOL (Company Name Search + yfinance)")
    print("=" * 70)
    print("✅ Type any company name (e.g. Apple, Tesla, Google, Reliance)")
    print("💡 Tool will auto-detect correct ticker symbol and analyze")
    print("=" * 70)

    while True:
        query = input("\nEnter company name or ticker (or 'exit'): ").strip()
        if query.lower() == "exit":
            print("\n👋 Exiting program. Stay profitable!\n")
            sys.exit(0)

        ticker = find_ticker_from_name(query)
        if not ticker:
            continue

        print("\n" + "=" * 70)
        print(f"Analyzing {ticker}")
        print("=" * 70)

        data, ticker_obj = get_stock_data(ticker)
        if data is None:
            print("⚠️ Failed to get data. Try another company.")
            continue

        show_company_info(ticker_obj, ticker)
        show_basic_stats(data)
        data = calculate_indicators(data)
        show_daily_return_stats(data)
        show_volatility(data, ticker)
        plot_stock_data(data, ticker)
        predict_next_close(data)
        plot_regression_trend(data, ticker)
        export_to_csv(data, ticker)

        print("\n✅ Analysis completed successfully for", ticker)
        print("-" * 70)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
