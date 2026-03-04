import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v9.0", layout="wide")
st.title("🛡️ Pro-Tracer: Dual-Benchmark Alpha Suite")

ticker = st.sidebar.text_input("Ticker Symbol", "BRITANNIA.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))

@st.cache_data
def get_alpha_data(symbol, start):
    # Fetching Stock, Sector Index, and Market Index
    s_df = yf.download(symbol, start=start)
    fmcg_df = yf.download("NIFTY_FMCG.NS", start=start)
    nifty_df = yf.download("^NSEI", start=start)
    
    # Flatten MultiIndex columns if present
    for d in [s_df, fmcg_df, nifty_df]:
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)

    df = pd.DataFrame(index=s_df.index)
    df['Close'] = s_df['Close']
    df['FMCG_Close'] = fmcg_df['Close']
    df['Nifty_Close'] = nifty_df['Close']
    df['Volume'] = s_df['Volume']
    return df

df = get_alpha_data(ticker, start_date)

if not df.empty:
    # --- RELATIVE STRENGTH CALCULATIONS ---
    # RS vs Nifty 50 (Market Benchmark)
    df['RS_Market'] = df['Close'] / df['Nifty_Close']
    df['RS_Market_Trend'] = ta.ema(df['RS_Market'], length=50)
    
    # RS vs Nifty FMCG (Sector Benchmark)
    df['RS_Sector'] = df['Close'] / df['FMCG_Close']
    df['RS_Sector_Trend'] = ta.ema(df['RS_Sector'], length=50)

    # GMMA & RSI Logic
    df['RSI'] = ta.rsi(df['Close'], length=13)
    df['ST_Avg'] = df['Close'].rolling(window=8).mean() 
    df['LT_Avg'] = df['Close'].rolling(window=45).mean()
    df['LT_Width'] = (df['Close'].rolling(window=30).max() - df['Close'].rolling(window=60).min())

    # --- ALPHA FILTERS ---
    # Buy only when Stock > Sector AND Sector is showing resilience
    df['Alpha_Buy'] = (df['RS_Sector'] > df['RS_Sector_Trend']) & (df['RS_Market'] > df['RS_Market_Trend'])

    # --- SIDEBAR STATUS ---
    st.sidebar.subheader("📊 Alpha Scorecard")
    is_leader = df['Alpha_Buy'].iloc[-1]
    st.sidebar.markdown(f"**Sector Leader:** {'✅ YES' if is_leader else '❌ NO'}")

    # --- BACKTESTER ---
    trades, in_trade, peak_p = [], False, 0
    for i in range(1, len(df)):
        curr_p, rsi_v = float(df['Close'].iloc[i]), df['RSI'].iloc[i]
        
        if not in_trade:
            # ALPHA ENTRY: GMMA + RSI + Sector Strength
            if (df['Alpha_Buy'].iloc[i] and rsi_v > 55 and df['ST_Avg'].iloc[i] > df['LT_Avg'].iloc[i]):
                in_trade, entry_date, entry_price = True, df.index[i], curr_p
                peak_p = curr_p
        elif in_trade:
            peak_p = max(peak_p, curr_p)
            # DUAL EXIT: RSI or 10% Trailing Stop
            if rsi_v < 45 or curr_p <= (peak_p * 0.90):
                in_trade = False
                pnl = ((curr_p - entry_price) / entry_price) * 100
                trades.append({"Entry": entry_date.date(), "Exit": df.index[i].date(), "P&L %": round(pnl, 2)})

    if trades:
        t_df = pd.DataFrame(trades)
        st.subheader("🏆 Strategy Performance (Sector Alpha Filter)")
        st.metric("Total Alpha ROI", f"{t_df['P&L %'].sum():.1f}%")
        st.dataframe(t_df, use_container_width=True)