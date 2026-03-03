import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def run_2d_optimization(ticker):
    # Download data once to save time
    df = yf.download(ticker, start="2020-01-01")
    df.columns = df.columns.get_level_values(0)
    
    results = []
    
    # Define ranges for the Grid Search
    # Using larger steps (e.g., 10) for speed, but you can refine this
    rsi_lengths = range(10, 201, 10) 
    buy_thresholds = range(40, 71, 5)

    for length in rsi_lengths:
        # Pre-calculate RSI for this length to avoid redundant work
        df['RSI'] = ta.rsi(df['Close'], length=length)
        
        for threshold in buy_thresholds:
            temp_df = df.copy()
            
            # Entry: RSI > Threshold | Exit: RSI < 40
            temp_df['Signal'] = 0
            temp_df.loc[temp_df['RSI'] > threshold, 'Signal'] = 1
            
            temp_df['Ret'] = temp_df['Close'].pct_change()
            temp_df['Strat_Ret'] = temp_df['Ret'] * temp_df['Signal'].shift(1)
            
            total_return = (1 + temp_df['Strat_Ret']).prod() - 1
            
            results.append({
                'RSI_Length': length,
                'Threshold': threshold,
                'Return': total_return * 100
            })
            
    return pd.DataFrame(results)

# --- Streamlit UI ---
st.title("Pro-Tracer: 2D Strategy Arena")

ticker = st.text_input("Ticker Symbol", "TRENT.NS")

if st.button("Generate Heatmap"):
    with st.spinner("Running Brute-Force Grid Search..."):
        opt_df = run_2d_optimization(ticker)
        
        # Reshape data for the heatmap
        pivot_table = opt_df.pivot(index="RSI_Length", columns="Threshold", values="Return")
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax)
        plt.title(f"ROI Heatmap: RSI Length vs Entry Threshold ({ticker})")
        
        st.pyplot(fig)