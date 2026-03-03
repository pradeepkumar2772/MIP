import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v3.4", layout="wide")

st.title("🛡️ Pro-Tracer: Deep-Cycle Optimizer Edition")
st.sidebar.header("Global Settings")

# --- Global Inputs ---
ticker = st.sidebar.text_input("Ticker Symbol", "TRENT.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(1999, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

mode = st.sidebar.radio("Select Module", ["Brute-Force Optimizer", "Trade Detailer"])

@st.cache_data
def get_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return data
    except Exception as e:
        st.error(f"Data error: {e}")
    return pd.DataFrame()

df_raw = get_data(ticker, start_date, end_date)

if df_raw.empty:
    st.warning("Awaiting data...")
else:
    # --- MODULE 1: DEEP-CYCLE OPTIMIZER ---
    if mode == "Brute-Force Optimizer":
        st.header("🔍 Macro RSI Optimizer (3 to 252)")
        st.info("Testing 50+ combinations of RSI speeds to find the 'Golden Cycle'.")
        
        if st.button("🚀 Start Deep Scan"):
            results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            
            # --- UPDATED RANGE: 3 to 252 with Step 5 for Speed ---
            rsi_range = range(3, 253, 5) 
            progress_bar = st.progress(0)
            
            for i, r_len in enumerate(rsi_range):
                progress_bar.progress((i + 1) / len(rsi_range))
                
                # 1. Calculate RSI for this length
                rsi = ta.rsi(df['Close'], length=r_len)
                
                # 2. Fast Simulation Logic
                sig = pd.Series(0, index=df.index)
                in_pos = False
                for j in range(1, len(df)):
                    # Entry: RSI > 60, Exit: RSI < 50
                    if not in_pos and rsi.iloc[j] > 60: in_pos = True
                    elif in_pos and rsi.iloc[j] < 50: in_pos = False
                    sig.iloc[j] = 1 if in_pos else 0
                
                # 3. Performance Math
                strat_ret = (df['Market_Ret'] * sig.shift(1)).fillna(0)
                cum_ret = (1 + strat_ret).cumprod()
                total_return = cum_ret.iloc[-1] - 1
                max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
                
                # 4. Result Logging
                results.append({
                    'RSI_Len': r_len, 
                    'ROI %': round(total_return * 100, 2), 
                    'Max_DD %': round(max_dd * 100, 2),
                    'Profit/DD': round(abs(total_return / max_dd), 2) if max_dd != 0 else 0
                })
                
            res_df = pd.DataFrame(results)
            st.success("Deep Scan Complete!")
            st.write("### Top Strategies (Ranked by ROI)")
            st.dataframe(res_df.sort_values('ROI %', ascending=False).head(20))

    # --- MODULE 2: TRADE DETAILER ---
    elif mode == "Trade Detailer":
        # ... [Keep your stable Trade Detailer code from v3.3 here] ...
        st.header("📜 Trade Detailer & Recovery Analysis")
        col1, col2, col3, col4 = st.columns(4)
        in_rsi = col1.number_input("RSI Look-back", value=14)
        vol_mult = col2.number_input("Vol Spike (x Avg)", value=1.5)
        vol_ma = col3.number_input("Vol Avg Period", value=20)
        stop_loss_pct = col4.number_input("Stop Loss %", value=5.0)
        
        # [Remaining logic for metrics, ledger with Prices, and chart generation]
        # (Refer to v3.3 for the detailed looping logic)