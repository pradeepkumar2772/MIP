import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st

def optimize_rsi_ema_crossover(ticker, rsi_range, ema_range):
    # 1. Fetch Data (Daily)
    df = yf.download(ticker, start="2022-01-01")
    df.columns = df.columns.get_level_values(0)
    
    results = []

    # 2. Brute Force Optimizer
    for r_len in rsi_range:
        # Calculate Base RSI
        rsi_series = ta.rsi(df['Close'], length=r_len)
        
        for e_len in ema_range:
            # Calculate EMA of the RSI (Signal Line)
            rsi_ema = ta.ema(rsi_series, length=e_len)
            
            # 3. Strategy Logic: Long when RSI > EMA, Exit when RSI < EMA
            # We use shift(1) to simulate entering at the Open of the next day
            signal = (rsi_series > rsi_ema).astype(int)
            
            returns = df['Close'].pct_change()
            strategy_returns = returns * signal.shift(1)
            
            # 4. Performance Metrics
            total_return = (1 + strategy_returns).prod() - 1
            
            # Count number of trades to check for over-trading
            trade_count = (signal.diff().abs() == 1).sum()
            
            results.append({
                'RSI_Len': r_len,
                'EMA_Len': e_len,
                'ROI %': total_return * 100,
                'Trades': trade_count
            })
            
    return pd.DataFrame(results)

# --- Streamlit UI ---
st.title("⚡ RSI vs. EMA Signal Optimizer")

ticker = st.text_input("Enter Ticker", "TRENT.NS")

# Define ranges: Testing broad RSI lengths vs specific EMA smoothing
rsi_vals = range(14, 101, 7)  # Example: 14 to 100 in steps of 7
ema_vals = range(5, 31, 5)    # Example: 5 to 30 in steps of 5

if st.button("Run EMA Optimization"):
    with st.spinner("Calculating Exponential Momentum..."):
        report = optimize_rsi_ema_crossover(ticker, rsi_vals, ema_vals)
        
        # Sort and Display
        report = report.sort_values(by='ROI %', ascending=False)
        st.write("### Top Optimized Parameters")
        st.dataframe(report.head(10))

        # Visualizing the ROI Distribution
        st.write("### Profitability Map (RSI vs ROI)")
        st.bar_chart(data=report, x='RSI_Len', y='ROI %')