import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import matplotlib.pyplot as plt

def run_3d_rsi_optimization(ticker):
    df = yf.download(ticker, start="2020-01-01")
    df.columns = df.columns.get_level_values(0)
    
    results = []
    
    # Grid Search Parameters
    rsi_range = range(10, 201, 10)   # RSI Length
    signal_range = range(5, 51, 5)    # Average Line Length

    for r_len in rsi_range:
        # Step 1: Calculate Base RSI
        rsi_series = ta.rsi(df['Close'], length=r_len)
        
        for s_len in signal_range:
            # Step 2: Calculate Average Line of that RSI
            rsi_sma = ta.sma(rsi_series, length=s_len)
            
            # Step 3: Strategy - Entry when RSI > RSI_SMA
            # We use vectorized logic for speed
            signal = (rsi_series > rsi_sma).astype(int)
            
            returns = df['Close'].pct_change()
            strat_returns = returns * signal.shift(1)
            
            total_return = (1 + strat_returns).prod() - 1
            
            results.append({
                'RSI_Length': r_len,
                'Signal_Length': s_len,
                'Return': total_return * 100
            })
            
    return pd.DataFrame(results)

# --- Streamlit UI ---
st.title("Pro-Tracer: 3D RSI + Average Line Optimizer")

ticker = st.text_input("Ticker Symbol", "TRENT.NS")

if st.button("Battle-Test Parameters"):
    with st.spinner("Crunching thousands of combinations..."):
        opt_df = run_3d_rsi_optimization(ticker)
        
        # Pivot for Heatmap
        pivot = opt_df.pivot(index="RSI_Length", columns="Signal_Length", values="Return")
        
        # Rendering the Heatmap (No Seaborn needed)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(pivot, cmap="RdYlGn", aspect='auto')
        
        # Formatting
        plt.colorbar(im, label='Total Return %')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        
        plt.xlabel("Average Line (Signal) Length")
        plt.ylabel("RSI Look-back Length")
        plt.title(f"3D Optimization: {ticker}")
        
        st.pyplot(fig)