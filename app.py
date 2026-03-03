import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="Pro-Tracer v2.2", layout="wide")
st.title("🛡️ Pro-Tracer: Triple-Parameter Optimizer")
st.sidebar.header("Global Settings")

ticker = st.sidebar.text_input("Ticker Symbol", "TRENT.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

@st.cache_data
def get_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    if not data.empty and isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

df_raw = get_data(ticker, start_date, end_date)

if df_raw.empty:
    st.warning("Please enter a valid ticker.")
else:
    st.header("🔍 Brute-Force: RSI + EMA + Stop Loss")
    st.info("Finding the combination that maximizes ROI while minimizing Drawdown.")

    # Optimization Ranges
    # We use steps (e.g., 10) to keep the web-app fast. Change to 1 for maximum precision.
    rsi_range = range(10, 101, 10) 
    ema_range = range(5, 31, 5)
    sl_range = [3, 5, 8, 12] # Testing 3%, 5%, 8%, and 12% Stop Losses

    if st.button("🚀 Run Triple Optimization"):
        results = []
        df = df_raw.copy()
        
        # Calculate Market Returns
        df['Ret'] = df['Close'].pct_change()
        
        progress_bar = st.progress(0)
        total_steps = len(rsi_range) * len(ema_range) * len(sl_range)
        current_step = 0

        for r_len in rsi_range:
            rsi_series = ta.rsi(df['Close'], length=r_len)
            
            for e_len in ema_range:
                rsi_ema = ta.ema(rsi_series, length=e_len)
                
                for sl_val in sl_range:
                    current_step += 1
                    # Simple simulation of the Stop Loss logic
                    # 1. Base Strategy (RSI > EMA)
                    signal = (rsi_series > rsi_ema).astype(int)
                    
                    # 2. Apply Returns
                    strat_ret = (df['Ret'] * signal.shift(1)).fillna(0)
                    
                    # 3. Stop Loss Filter: If a daily loss exceeds the SL, we cap it.
                    # This is a simplified "Daily Stop" for the optimizer.
                    # For precise trade-by-trade stops, use the 'Trade Detailer' module.
                    strat_ret = strat_ret.apply(lambda x: max(x, -sl_val/100))
                    
                    cum_ret = (1 + strat_ret).cumprod()
                    total_return = cum_ret.iloc[-1] - 1
                    max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
                    
                    results.append({
                        'RSI': r_len,
                        'EMA': e_len,
                        'StopLoss %': sl_val,
                        'ROI %': round(total_return * 100, 2),
                        'Max_DD %': round(max_dd * 100, 2),
                        'Profit/DD Ratio': round(abs(total_return/max_dd), 2) if max_dd != 0 else 0
                    })
                    
            progress_bar.progress(current_step / total_steps)

        res_df = pd.DataFrame(results)
        st.success("Optimization Complete!")
        
        st.write("### Top 15 Combinations (Ranked by Profit/DD Ratio)")
        # We sort by Profit/DD Ratio because that finds the most "Stable" strategy
        st.dataframe(res_df.sort_values('Profit/DD Ratio', ascending=False).head(15))