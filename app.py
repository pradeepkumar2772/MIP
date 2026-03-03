import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v7.0", layout="wide")

st.title("🛡️ Pro-Tracer: Relative Strength (Nifty 50) Suite")

ticker = st.sidebar.text_input("Ticker Symbol", "BRITANNIA.NS")
index_ticker = "^NSEI" # Standard YFinance symbol for Nifty 50
start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

@st.cache_data
def get_dual_data(stock, index, start, end):
    # Fetching both Stock and Nifty 50
    s_data = yf.download(stock, start=start, end=end)
    idx_data = yf.download(index, start=start, end=end)
    
    if s_data.empty or idx_data.empty: return pd.DataFrame()
    
    # Align columns
    if isinstance(s_data.columns, pd.MultiIndex):
        s_data.columns = s_data.columns.get_level_values(0)
    if isinstance(idx_data.columns, pd.MultiIndex):
        idx_data.columns = idx_data.columns.get_level_values(0)
        
    combined = pd.DataFrame(index=s_data.index)
    combined['Stock_Close'] = s_data['Close']
    combined['Index_Close'] = idx_data['Close']
    combined['Volume'] = s_data['Volume']
    return combined

df = get_dual_data(ticker, index_ticker, start_date, end_date)

if not df.empty:
    # --- CALCULATE RELATIVE STRENGTH (RS) ---
    # RS Ratio = Stock / Index
    df['RS_Ratio'] = df['Stock_Close'] / df['Index_Close']
    # RS EMA to see the trend of outperformance
    df['RS_Trend'] = ta.ema(df['RS_Ratio'], length=50)
    
    # Existing Indicators
    df['RSI'] = ta.rsi(df['Stock_Close'], length=13)
    df['ST_Avg'] = df['Stock_Close'].rolling(window=8).mean() 
    df['LT_Avg'] = df['Stock_Close'].rolling(window=45).mean()

    # --- UPDATED ENTRY LOGIC ---
    # We only enter if Stock is outperforming Nifty (RS_Ratio > RS_Trend)
    df['Is_Outperforming'] = df['RS_Ratio'] > df['RS_Trend']
    
    st.subheader("🔍 Market Performance Filter")
    latest_rs = df['Is_Outperforming'].iloc[-1]
    color = "green" if latest_rs else "red"
    st.markdown(f"**Current Status:** <span style='color:{color}'>{'OUTPERFORMING' if latest_rs else 'UNDERPERFORMING'}</span> Nifty 50", unsafe_allow_html=True)

    # --- TRADE ENGINE ---
    # (Applying your existing GMMA and RSI logic + RS Filter)
    trades, in_trade = [], False
    for i in range(1, len(df)):
        if not in_trade:
            # Entry conditions
            if (df['Is_Outperforming'].iloc[i] and 
                df['RSI'].iloc[i] > 55 and 
                df['ST_Avg'].iloc[i] > df['LT_Avg'].iloc[i]):
                in_trade, entry_date, entry_price = True, df.index[i], df['Stock_Close'].iloc[i]
        elif in_trade:
            if df['RSI'].iloc[i] < 45: # RSI Exit
                in_trade = False
                pnl = ((df['Stock_Close'].iloc[i] - entry_price) / entry_price) * 100
                trades.append({"Date": df.index[i].date(), "P&L %": round(pnl, 2)})

    if trades:
        st.write("### 📋 Backtest Results with RS Filter")
        st.dataframe(pd.DataFrame(trades), use_container_width=True)