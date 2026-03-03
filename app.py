import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt

# 1. Streamlit UI
st.title("Pro-Tracer: RSI Backtest")
ticker = st.text_input("Enter Ticker (e.g., RELIANCE.NS)", value="RELIANCE.NS")

# 2. Data Download
data = yf.download(ticker, start="2020-01-01")

if not data.empty:
    # 3. RSI Calculation
    data['RSI'] = ta.rsi(data['Close'], length=14)

    # 4. Strategy Logic
    data['Signal'] = 0
    data.loc[data['RSI'] > 50, 'Signal'] = 1
    data.loc[data['RSI'] < 40, 'Signal'] = 0

    # 5. Returns Calculation
    data['Market_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Market_Return'] * data['Signal'].shift(1)
    data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).fillna(0).cumprod()
    data['Cumulative_Market'] = (1 + data['Market_Return']).fillna(0).cumprod()

    # 6. Plotting (The Streamlit Way)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Top Plot: Performance
    ax1.plot(data['Cumulative_Strategy'], label='RSI Strategy', color='green')
    ax1.plot(data['Cumulative_Market'], label='Buy & Hold', color='gray', alpha=0.5)
    ax1.set_title(f"Equity Curve: {ticker}")
    ax1.legend()
    ax1.grid(True)

    # Bottom Plot: RSI
    ax2.plot(data['RSI'], color='purple')
    ax2.axhline(50, color='black', linestyle='--', alpha=0.5)
    ax2.axhline(70, color='red', linestyle=':', alpha=0.3)
    ax2.axhline(30, color='blue', linestyle=':', alpha=0.3)
    ax2.set_title("RSI Oscillator")
    ax2.grid(True)

    plt.tight_layout()

    # INSTEAD OF plt.show(), use st.pyplot()
    st.pyplot(fig)

    # 7. Metrics
    total_ret = (data['Cumulative_Strategy'].iloc[-1] - 1) * 100
    st.metric("Total Strategy Return", f"{total_ret:.2f}%")
else:
    st.error("No data found for this ticker.")