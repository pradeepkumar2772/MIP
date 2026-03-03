import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="Pro-Tracer v2.0", layout="wide")

st.title("🛡️ Pro-Tracer: The Quantitative Engine")
st.sidebar.header("Control Panel")

# 1. Shared Inputs
ticker = st.sidebar.text_input("Ticker Symbol", "RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(1990, 1, 1), min_value=datetime.date(1990, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date(2026, 3, 3))

# Navigation
mode = st.sidebar.radio("Select Module", ["Brute-Force Optimizer", "Trade Detailer"])

# --- DATA DOWNLOADER ---
@st.cache_data
def get_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    if not data.empty:
        data.columns = data.columns.get_level_values(0)
    return data

df_raw = get_data(ticker, start_date, end_date)

if df_raw.empty:
    st.error("No data found. Try a different ticker or date.")
else:
    # --- MODULE 1: BRUTE-FORCE OPTIMIZER ---
    if mode == "Brute-Force Optimizer":
        st.header("🔍 Parameter Optimizer (RSI vs Average Line)")
        st.write("Varying RSI (3-252) and EMA (3-50) to find the best Risk/Reward.")
        
        if st.button("🚀 Start Optimization"):
            results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            
            # For speed in web-app, we use a step of 2 for RSI
            rsi_range = range(3, 253, 2)
            ema_range = range(3, 51, 2)
            
            progress_bar = st.progress(0)
            total_steps = len(rsi_range)

            for i, r_len in enumerate(rsi_range):
                progress_bar.progress(i / total_steps)
                rsi_series = ta.rsi(df['Close'], length=r_len)
                
                for e_len in ema_range:
                    rsi_ema = ta.ema(rsi_series, length=e_len)
                    
                    # Signal & Returns
                    signal = (rsi_series > rsi_ema).astype(int)
                    strat_ret = (df['Market_Ret'] * signal.shift(1)).fillna(0)
                    cum_ret = (1 + strat_ret).cumprod()
                    
                    # Metrics
                    total_return = (cum_ret.iloc[-1] - 1)
                    running_max = cum_ret.cummax()
                    drawdown = (cum_ret - running_max) / running_max
                    max_dd = drawdown.min()
                    
                    results.append({
                        'RSI_Len': r_len,
                        'EMA_Len': e_len,
                        'ROI_%': round(total_return * 100, 2),
                        'Max_DD_%': round(max_dd * 100, 2),
                        'Risk_Score': round(abs(total_return/max_dd), 2) if max_dd != 0 else 0
                    })
            
            res_df = pd.DataFrame(results)
            st.success("Optimization Complete!")
            st.write("### Top 20 Parameters by ROI")
            st.dataframe(res_df.sort_values('ROI_%', ascending=False).head(20))

    # --- MODULE 2: TRADE DETAILER ---
    elif mode == "Trade Detailer":
        st.header("📜 Individual Trade Ledger")
        
        col1, col2 = st.columns(2)
        in_rsi = col1.number_input("Input RSI Length", value=14)
        in_ema = col2.number_input("Input EMA (Average Line) Length", value=9)
        
        if st.button("📖 Generate Trade Book"):
            df = df_raw.copy()
            df['RSI'] = ta.rsi(df['Close'], length=in_rsi)
            df['RSI_EMA'] = ta.ema(df['RSI'], length=in_ema)
            df['Signal'] = (df['RSI'] > df['RSI_EMA']).astype(int)
            df['Change'] = df['Signal'].diff()

            trades = []
            in_trade = False
            
            for i in range(1, len(df)):
                # Buy
                if df['Change'].iloc[i] == 1 and not in_trade:
                    in_trade = True
                    entry_date = df.index[i]
                    entry_price = df['Close'].iloc[i]
                # Sell
                elif df['Change'].iloc[i] == -1 and in_trade:
                    in_trade = False
                    exit_date = df.index[i]
                    exit_price = df['Close'].iloc[i]
                    pnl = ((exit_price - entry_price) / entry_price) * 100
                    
                    trades.append({
                        "Entry Date": entry_date.date(),
                        "Exit Date": exit_date.date(),
                        "Entry Price": round(float(entry_price), 2),
                        "Exit Price": round(float(exit_price), 2),
                        "P&L %": round(float(pnl), 2)
                    })

            if trades:
                t_df = pd.DataFrame(trades)
                st.write(f"### Trade Log for RSI {in_rsi} / EMA {in_ema}")
                st.dataframe(t_df)
                
                # Visual summary
                st.subheader("Performance Summary")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Trades", len(t_df))
                c2.metric("Win Rate", f"{(t_df['P&L %'] > 0).mean()*100:.1f}%")
                c3.metric("Avg Trade %", f"{t_df['P&L %'].mean():.2f}%")
            else:
                st.warning("No trades found for these settings.")