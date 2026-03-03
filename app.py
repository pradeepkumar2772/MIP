import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v4.0", layout="wide")

st.title("🛡️ Pro-Tracer: 3D Matrix Optimizer")
st.sidebar.header("Global Settings")

ticker = st.sidebar.text_input("Ticker Symbol", "TRENT.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(1999, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

mode = st.sidebar.radio("Select Module", ["Matrix Optimizer", "Trade Detailer"])

@st.cache_data
def get_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    if not data.empty and isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

df_raw = get_data(ticker, start_date, end_date)

if df_raw.empty:
    st.warning("Awaiting data...")
else:
    if mode == "Matrix Optimizer":
        st.header("🧪 RSI Entry/Exit Matrix Scan")
        st.info("Finding the best combination of RSI Length, Entry Level, and Exit Level.")
        
        # Matrix Controls
        c1, c2 = st.columns(2)
        opt_rsi_len = c1.slider("Fixed RSI Length for Matrix", 3, 252, 14)
        
        if st.button("🚀 Run Matrix Scan"):
            matrix_results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            rsi = ta.rsi(df['Close'], length=opt_rsi_len)
            
            # Threshold Ranges
            entry_range = range(55, 81, 5) # 55, 60, 65, 70, 75, 80
            exit_range = range(35, 61, 5)  # 35, 40, 45, 50, 55, 60
            
            progress_bar = st.progress(0)
            total_steps = len(entry_range) * len(exit_range)
            step = 0
            
            for ent in entry_range:
                for ext in exit_range:
                    if ext >= ent: continue # Exit cannot be higher than entry
                    
                    step += 1
                    progress_bar.progress(step / total_steps)
                    
                    # Simulation
                    sig = pd.Series(0, index=df.index)
                    in_pos = False
                    for j in range(1, len(df)):
                        if not in_pos and rsi.iloc[j] > ent: in_pos = True
                        elif in_pos and rsi.iloc[j] < ext: in_pos = False
                        sig.iloc[j] = 1 if in_pos else 0
                    
                    strat_ret = (df['Market_Ret'] * sig.shift(1)).fillna(0)
                    cum_ret = (1 + strat_ret).cumprod()
                    total_roi = (cum_ret.iloc[-1] - 1) * 100
                    
                    matrix_results.append({'Entry': ent, 'Exit': ext, 'ROI': total_roi})
            
            res_df = pd.DataFrame(matrix_results)
            pivot_df = res_df.pivot(index="Entry", columns="Exit", values="ROI")
            
            # --- Heatmap Visualization ---
            st.write("### 🌡️ ROI Heatmap (Entry vs Exit)")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap="RdYlGn", ax=ax)
            st.pyplot(fig)
            
            st.write("### 🏆 Top Matrix Combinations")
            st.dataframe(res_df.sort_values('ROI', ascending=False))

    elif mode == "Trade Detailer":
        # Standard v3.3 Detailer logic here...
        st.info("Use the parameters from the Matrix Optimizer here to see full trade logs.")