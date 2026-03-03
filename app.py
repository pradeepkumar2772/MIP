import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np

# --- Streamlit UI Setup ---
st.title("🛡️ Pro-Tracer: Legacy Optimizer (1990+)")
st.markdown("### RSI vs EMA Brute-Force + Risk Metrics")

# 1. User Inputs
ticker = st.text_input("Ticker Symbol", "RELIANCE.NS")

col1, col2 = st.columns(2)
# Updated to allow 1990
start_date = col1.date_input("Start Date", datetime.date(1990, 1, 1), min_value=datetime.date(1990, 1, 1))
end_date = col2.date_input("End Date", datetime.date(2026, 3, 1))

# 2. Optimization Parameters
rsi_min, rsi_max = 3, 252
ema_min, ema_max = 3, 50

if st.button("🚀 Start Full Optimization"):
    with st.spinner("Downloading and processing decades of data..."):
        df = yf.download(ticker, start=start_date, end=end_date)
        df.columns = df.columns.get_level_values(0)
    
    if df.empty:
        st.error("No data found. Note: Many NSE stocks only have data from the late 90s.")
    else:
        results = []
        progress_bar = st.progress(0)
        
        # Calculate market returns once
        df['Market_Ret'] = df['Close'].pct_change()
        
        total_steps = (rsi_max - rsi_min + 1)
        for i, r_len in enumerate(range(rsi_min, rsi_max + 1)):
            if i % 15 == 0:
                progress_bar.progress(i / total_steps)
                
            rsi_series = ta.rsi(df['Close'], length=r_len)
            
            for e_len in range(ema_min, ema_max + 1):
                rsi_ema = ta.ema(rsi_series, length=e_len)
                
                # Signal: 1 if RSI > EMA else 0
                signal = (rsi_series > rsi_ema).astype(int)
                strat_ret = (df['Market_Ret'] * signal.shift(1)).fillna(0)
                
                # Cumulative Returns
                cum_ret = (1 + strat_ret).cumprod()
                total_return = cum_ret.iloc[-1] - 1
                
                # Max Drawdown Calculation
                # 
                running_max = cum_ret.cummax()
                drawdown = (cum_ret - running_max) / running_max
                max_dd = drawdown.min()
                
                results.append({
                    'RSI_Len': r_len,
                    'EMA_Len': e_len,
                    'ROI_%': total_return * 100,
                    'Max_DD_%': max_dd * 100
                })
        
        progress_bar.progress(1.0)
        
        # 4. Display Results
        res_df = pd.DataFrame(results)
        # We name the index column to make it clear
        res_df.index.name = "Rank_ID"
        
        best = res_df.loc[res_df['ROI_%'].idxmax()]
        
        st.success(f"Best ROI: {best['ROI_%']:.2f}% | Associated Max Drawdown: {best['Max_DD_%']:.2f}%")
        
        # Sort by ROI and show
        st.write("### Top Parameters (Sorted by ROI)")
        st.dataframe(res_df.sort_values('ROI_%', ascending=False).head(20))