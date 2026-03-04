import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v8.1", layout="wide")

st.title("🛡️ Pro-Tracer: GMMA Ribbon Color Analyzer")

ticker = st.sidebar.text_input("Ticker Symbol", "BRITANNIA.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

@st.cache_data
def get_gmma_with_color(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # EMAs (Standard Guppy Periods)
    st_p = [3, 5, 8, 10, 12, 15]
    lt_p = [30, 35, 40, 45, 50, 60]
    
    for p in st_p: df[f'ST_{p}'] = ta.ema(df['Close'], length=p)
    for p in lt_p: df[f'LT_{p}'] = ta.ema(df['Close'], length=p)
    
    # Min/Max for Expansion Logic
    df['ST_Min'] = df[[f'ST_{p}' for p in st_p]].min(axis=1)
    df['LT_Max'] = df[[f'LT_{p}' for p in lt_p]].max(axis=1)
    df['LT_Min'] = df[[f'LT_{p}' for p in lt_p]].min(axis=1)
    
    # Expansion Width
    df['LT_Width'] = (df['LT_Max'] - df['LT_Min']) / df['LT_Min']
    return df

df = get_gmma_with_color(ticker, start_date, end_date)

if not df.empty:
    trades, in_trade = [], False
    
    for i in range(1, len(df)):
        st_min, lt_max = df['ST_Min'].iloc[i], df['LT_Max'].iloc[i]
        lt_w, prev_lt_w = df['LT_Width'].iloc[i], df['LT_Width'].iloc[i-1]
        
        if not in_trade:
            # Entry: Short-term > Long-term
            if st_min > lt_max:
                in_trade = True
                # Color Labeling:
                # Strong Green = Width Increasing
                # Recovering Yellow = Width Flat or Small
                state = "🟢 Strong Expansion" if lt_w > prev_lt_w else "🟡 Crossover"
                trades.append({"Entry": df.index[i].date(), "State": state, "entry_p": df['Close'].iloc[i]})
        
        elif in_trade:
            # Exit: Ribbon Break
            if st_min < lt_max:
                in_trade = False
                pnl = ((df['Close'].iloc[i] - trades[-1]['entry_p']) / trades[-1]['entry_p']) * 100
                trades[-1]['Exit'] = df.index[i].date()
                trades[-1]['ROI %'] = round(pnl, 2)

    if trades:
        st.write("### 🏆 Ribbon Strategy Performance")
        st.table(pd.DataFrame(trades).drop(columns=['entry_p']))