import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v8.3", layout="wide")

st.title("🛡️ Pro-Tracer: Volume-Confirmed GMMA Squeeze")

# --- SIDEBAR: SETTINGS & LIVE ALERTS ---
st.sidebar.header("Strategy Settings")
ticker = st.sidebar.text_input("Ticker Symbol", "BRITANNIA.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))

c1, c2 = st.sidebar.columns(2)
comp_threshold = c1.slider("Squeeze %", 0.1, 2.0, 0.8)
rvol_threshold = c2.number_input("Min RVOL (x)", value=1.5, step=0.1)

@st.cache_data
def get_confirmed_data(symbol, start):
    df = yf.download(symbol, start=start)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # GMMA Logic
    st_p, lt_p = [3, 5, 8, 10, 12, 15], [30, 35, 40, 45, 50, 60]
    for p in st_p: df[f'ST_{p}'] = ta.ema(df['Close'], length=p)
    for p in lt_p: df[f'LT_{p}'] = ta.ema(df['Close'], length=p)
    
    # Width & Volume Logic
    df['LT_Max'] = df[[f'LT_{p}' for p in lt_p]].max(axis=1)
    df['LT_Min'] = df[[f'LT_{p}' for p in lt_p]].min(axis=1)
    df['LT_Width_Pct'] = ((df['LT_Max'] - df['LT_Min']) / df['LT_Min']) * 100
    df['ST_Min'] = df[[f'ST_{p}' for p in st_p]].min(axis=1)
    
    # RVOL Calculation (Volume vs 20-day MA)
    df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
    df['RVOL'] = df['Volume'] / df['Vol_MA']
    return df

df = get_confirmed_data(ticker, start_date)

if not df.empty:
    # --- LIVE VOLUME-CONFIRMED ALERT ---
    curr_w = df['LT_Width_Pct'].iloc[-1]
    curr_rvol = df['RVOL'].iloc[-1]
    is_sqz = curr_w <= comp_threshold
    
    st.sidebar.subheader("📢 Institutional Monitor")
    if is_sqz:
        st.sidebar.warning(f"⚠️ SQUEEZE: Width {curr_w:.2f}%. Volume: {curr_rvol:.2f}x")
    elif df['ST_Min'].iloc[-1] > df['LT_Max'].iloc[-1] and curr_rvol > rvol_threshold:
        st.sidebar.success(f"🚀 BREAKOUT CONFIRMED: High Vol ({curr_rvol:.2f}x) detected!")

    # --- CONFIRMED BREAKOUT BACKTESTER ---
    trades, in_trade = [], False
    for i in range(1, len(df)):
        st_min, lt_max = df['ST_Min'].iloc[i], df['LT_Max'].iloc[i]
        rvol = df['RVOL'].iloc[i]
        
        if not in_trade:
            # ENTRY: Ribbon Breakout + Institutional Volume
            if st_min > lt_max and rvol >= rvol_threshold:
                in_trade = True
                was_sqz = any(df['LT_Width_Pct'].iloc[i-15:i] <= comp_threshold)
                trades.append({
                    "Date": df.index[i].date(),
                    "Entry P": round(df['Close'].iloc[i], 2),
                    "Entry RVOL": round(rvol, 2),
                    "Context": "💎 Squeeze + Vol" if was_sqz else "📈 Momentum + Vol"
                })
        elif in_trade:
            if st_min < lt_max:
                in_trade = False
                pnl = ((df['Close'].iloc[i] - trades[-1]['Entry P']) / trades[-1]['Entry P']) * 100
                trades[-1]['Exit Date'] = df.index[i].date()
                trades[-1]['ROI %'] = round(pnl, 2)

    if trades:
        st.write("### 🏆 Confirmed Trade History")
        st.table(pd.DataFrame(trades))