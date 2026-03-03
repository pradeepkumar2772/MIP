import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v6.0", layout="wide")

st.title("🛡️ Pro-Tracer: GMMA + RSI Institutional Engine")
st.sidebar.header("Global Settings")

ticker = st.sidebar.text_input("Ticker Symbol", "BRITANNIA.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(1999, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

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
    st.header("📜 GMMA Performance Audit")
    
    # --- Input Parameters ---
    c1, c2, c3, c4 = st.columns(4)
    rsi_len = c1.number_input("RSI Length", value=13)
    rsi_ent = c2.slider("RSI Entry Threshold", 40, 70, 55)
    rsi_ext = c3.slider("RSI Exit Threshold", 30, 60, 45)
    tsl_pct = c4.number_input("Trailing SL %", value=10.0)

    # --- Indicators ---
    df = df_raw.copy()
    df['RSI'] = ta.rsi(df['Close'], length=rsi_len)
    
    # GMMA Short-Term Group
    short_emas = [3, 5, 8, 10, 12, 15]
    for ema in short_emas:
        df[f'EMA_{ema}'] = ta.ema(df['Close'], length=ema)
    df['ST_Avg'] = df[[f'EMA_{e}' for e in short_emas]].mean(axis=1)

    # GMMA Long-Term Group
    long_emas = [30, 35, 40, 45, 50, 60]
    for ema in long_emas:
        df[f'EMA_{ema}'] = ta.ema(df['Close'], length=ema)
    df['LT_Avg'] = df[[f'EMA_{e}' for e in long_emas]].mean(axis=1)
    
    # Global Trend Filter
    df['EMA_200'] = ta.ema(df['Close'], length=200)

    # --- Backtesting Logic ---
    trades = []
    in_trade = False
    peak_price = 0

    for i in range(1, len(df)):
        curr_p = float(df['Close'].iloc[i])
        st_avg = df['ST_Avg'].iloc[i]
        lt_avg = df['LT_Avg'].iloc[i]
        rsi_val = df['RSI'].iloc[i]
        ema_200 = df['EMA_200'].iloc[i]
        
        if not in_trade:
            # ENTRY: ST-Ribbon above LT-Ribbon + RSI Signal + Above 200 EMA
            if st_avg > lt_avg and rsi_val > rsi_ent and curr_p > ema_200:
                in_trade, entry_date, entry_price = True, df.index[i], curr_p
                peak_price = curr_p
        
        elif in_trade:
            peak_price = max(peak_price, curr_p)
            tsl_trigger = curr_p <= (peak_price * (1 - tsl_pct / 100))
            rsi_exit = rsi_val < rsi_ext
            
            if tsl_trigger or rsi_exit:
                in_trade = False
                pnl = ((curr_p - entry_price) / entry_price) * 100
                trades.append({
                    "Entry Date": entry_date.date(),
                    "Exit Date": df.index[i].date(),
                    "Entry P": round(entry_price, 2),
                    "Exit P": round(curr_p, 2),
                    "P&L %": round(pnl, 2),
                    "Reason": "Trailing SL" if tsl_trigger else "RSI Exit"
                })

    if trades:
        t_df = pd.DataFrame(trades)
        daily_rets = pd.Series(0.0, index=df.index)
        for _, row in t_df.iterrows():
            daily_rets.loc[pd.to_datetime(row['Exit Date'])] = row['P&L %'] / 100
        cum_strategy = (1 + daily_rets).cumprod()
        
        # --- Scorecard ---
        st.subheader("📊 Institutional Scorecard")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final ROI", f"{(cum_strategy.iloc[-1]-1)*100:.1f}%")
        m2.metric("Win Rate", f"{(len(t_df[t_df['P&L %'] > 0])/len(t_df))*100:.1f}%")
        m3.metric("Max Drawdown", f"{((cum_strategy - cum_strategy.cummax())/cum_strategy.cummax()).min()*100:.2f}%")
        m4.metric("Total Trades", len(t_df))
        
        st.line_chart(100000 * cum_strategy)
        st.dataframe(t_df, use_container_width=True)
    else:
        st.warning("No trades triggered with GMMA + RSI filters.")