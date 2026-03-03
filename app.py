import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib
import matplotlib.pyplot as plt

# --- FIX: Force an interactive window (use 'TkAgg' or 'Qt5Agg') ---
# If one doesn't work, try the other. 
try:
    matplotlib.use('TkAgg') 
except:
    matplotlib.use('Qt5Agg')

# 1. Download Data
ticker = "RELIANCE.NS"
print(f"Downloading data for {ticker}...")
data = yf.download(ticker, start="2020-01-01")

# --- DATA CHECK ---
if data.empty:
    print("Error: No data downloaded. Check your internet or ticker symbol.")
else:
    print("Data received! Calculating RSI...")

    # 2. Calculate RSI
    data['RSI'] = ta.rsi(data['Close'], length=14)

    # 3. Simple Momentum Strategy
    data['Signal'] = 0
    data.loc[data['RSI'] > 50, 'Signal'] = 1  # Bullish
    data.loc[data['RSI'] < 40, 'Signal'] = 0  # Exit

    # 4. Calculate Strategy Performance
    data['Market_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Market_Return'] * data['Signal'].shift(1)

    data['Cumulative_Market'] = (1 + data['Market_Return']).fillna(0).cumprod()
    data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).fillna(0).cumprod()

    # 5. Create the Plot
    plt.figure(figsize=(12, 7))
    
    # Top Plot: Strategy vs Market
    plt.subplot(2, 1, 1)
    plt.plot(data['Cumulative_Strategy'], label='RSI Strategy (Pro-Tracer)', color='green', lw=2)
    plt.plot(data['Cumulative_Market'], label='Buy & Hold', color='gray', alpha=0.5)
    plt.title(f"RSI Momentum Performance: {ticker}")
    plt.legend()
    plt.grid(True)

    # Bottom Plot: RSI Levels
    plt.subplot(2, 1, 2)
    plt.plot(data['RSI'], color='purple', lw=1)
    plt.axhline(50, color='black', linestyle='--')
    plt.axhline(70, color='red', alpha=0.3)
    plt.axhline(30, color='blue', alpha=0.3)
    plt.fill_between(data.index, 30, 70, color='purple', alpha=0.05)
    plt.title("RSI Levels (14 period)")
    plt.grid(True)

    plt.tight_layout()
    
    print("Displaying chart... look for a new window!")
    plt.show() # <--- This is the most important line!