import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v2.6", layout="wide")

st.title("🛡️ Pro-Tracer: The Quad-Optimizer Engine")
st.sidebar.header("Global Settings")

# --- Global Inputs ---
ticker = st.sidebar.text_input("Ticker Symbol", "TRENT.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(2010, 1, 1), min_value=datetime.date(1990, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

mode = st.sidebar.radio("Select Module", ["Quad-Optimizer", "Trade Detailer"])

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
    # --- MODULE 1: QUAD-OPTIMIZER ---
    if mode == "Quad-Optimizer":
        st.header("🔍 Brute-Force: RSI + Signal + StopLoss + Trend Filter")
        st.info("Finding the 'Golden Guardrail' to crush that -47% Drawdown.")
        
        if st.button("🚀 Start Deep Optimization"):
            results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            
            # Optimization Ranges (Balanced for speed)
            rsi_range = range(10, 71, 10)     # RSI Lengths
            sig_range = range(5, 21, 5)      # RSI Signal EMA
            sl_range = [5, 8, 12]            # Hard Stop Loss %
            trend_range = [0, 50, 100, 200]  # Trend Filter (0 = Disabled)
            
            progress_bar = st.progress(0)
            total_steps = len(rsi_range) * len(sig_range) * len(sl_range) * len(trend_range)
            step_count = 0

            for r_len in rsi_range:
                rsi_series = ta.rsi(df['Close'], length=r_len)
                for s_len in sig_range:
                    sig_ema = ta.ema(rsi_series, length=s_len)
                    for tr_len in trend_range:
                        # Pre-calculate Trend Filter
                        if tr_len > 0:
                            trend_line = ta.ema(df['Close'], length=tr_len)
                            trend_ok = (df['Close'] > trend_line)
                        else:
                            trend_ok = True
                            
                        for sl_val in sl_range:
                            # 1. Generate Signal
                            raw_signal = (rsi_series > sig_ema) & trend_ok
                            
                            # 2. Simple Strategy Simulation
                            strat_ret = (df['Market_Ret'] * raw_signal.shift(1)).fillna(0)
                            
                            # 3. Apply Stop Loss (Capping daily downside)
                            strat_ret = strat_ret.apply(lambda x: max(x, -sl_val/100))
                            
                            cum_ret = (1 + strat_ret).cumprod()
                            total_return = cum_ret.iloc[-1] - 1
                            max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
                            
                            results.append({
                                'RSI': r_len, 'Sig': s_len, 'Trend': tr_len, 'SL %': sl_val,
                                'ROI %': round(total_return * 100, 2),
                                'Max_DD %': round(max_dd * 100, 2),
                                'Profit/DD': round(abs(total_return/max_dd), 2) if max_dd != 0 else 0
                            })
                            step_count += 1
                progress_bar.progress(step_count / total_steps)

            res_df = pd.DataFrame(results)
            st.success("Optimization Complete!")
            st.write("### Top Strategies (Ranked by Profit/DD Ratio)")
            st.dataframe(res_df.sort_values('Profit/DD', ascending=False).head(20))

    # --- MODULE 2: TRADE DETAILER ---
    elif mode == "Trade Detailer":
        # ... [Keep your stable Trade Detailer code from v2.5 here] ...
        st.header("📜 Detailer with Trend Confirmation")
        col1, col2, col3, col4 = st.columns(4)
        in_rsi = col1.number_input("RSI", value=14)
        in_ema = col2.number_input("Signal EMA", value=9)
        sl_pct = col3.number_input("Stop Loss %", value=5.0)
        tr_ema = col4.number_input("Trend EMA (0=Off)", value=200)
        
        # [The rest of the logic follows the detailed trade-by-trade loops as before]
        # (Refer to v2.5 for the full looping logic to generate the Quant Scorecard)