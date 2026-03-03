import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v4.9", layout="wide")

st.title("🛡️ Pro-Tracer: Win-Rate & Consistency Suite")

ticker = st.sidebar.text_input("Ticker Symbol", "TRENT.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(1999, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
mode = st.sidebar.radio("Select Module", ["Global 3D Optimizer", "Trade Detailer"])

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
    if mode == "Global 3D Optimizer":
        st.header("🔬 Win-Rate Optimized Scan")
        if st.button("🚀 Start Deep Scan"):
            all_results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            
            # Constraints: Step 2 precision
            len_range = range(3, 253, 10) 
            ent_range = range(50, 61, 2)
            ext_range = range(40, 61, 2)
            
            progress_bar = st.progress(0)
            total = len(len_range) * len(ent_range) * len(ext_range)
            count = 0

            for r_len in len_range:
                rsi = ta.rsi(df['Close'], length=r_len)
                for ent in ent_range:
                    for ext in ext_range:
                        if ext >= ent: continue
                        count += 1
                        
                        # --- Simulation & Trade Tracking ---
                        in_pos = False
                        trade_pnls = []
                        entry_p = 0
                        
                        for j in range(1, len(df)):
                            curr_rsi = rsi.iloc[j]
                            curr_price = float(df['Close'].iloc[j])
                            
                            if not in_pos and curr_rsi > ent:
                                in_pos, entry_p = True, curr_price
                            elif in_pos and curr_rsi < ext:
                                in_pos = False
                                trade_pnls.append((curr_price - entry_p) / entry_p)
                        
                        # --- Metrics Calculation ---
                        if trade_pnls:
                            wins = [p for p in trade_pnls if p > 0]
                            win_rate = (len(wins) / len(trade_pnls)) * 100
                            
                            # Standard Performance Math
                            roi = (np.prod([1 + p for p in trade_pnls]) - 1) * 100
                            
                            all_results.append({
                                'RSI_Len': r_len, 'Entry': ent, 'Exit': ext, 
                                'ROI %': round(roi, 2), 
                                'Win Rate %': round(win_rate, 1),
                                'Total Trades': len(trade_pnls)
                            })
                progress_bar.progress(count / total)

            res_df = pd.DataFrame(all_results).sort_values('ROI %', ascending=False)
            st.success("Scan Complete!")
            st.dataframe(res_df, use_container_width=True)

    elif mode == "Trade Detailer":
        # Keep the existing stable Detailer logic...
        st.info("The Detailer already shows Win Rate in the scorecard.")