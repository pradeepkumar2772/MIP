import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v8.2", layout="wide")

st.title("🛡️ Pro-Tracer: GMMA Compression & Breakout Suite")

# --- SIDEBAR: SETTINGS & LIVE ALERTS ---
st.sidebar.header("Strategy Settings")
ticker = st.sidebar.text_input("Ticker Symbol", "BRITANNIA.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
compression_threshold = st.sidebar.slider("Compression Sensitivity", 0.1, 2.0, 0.8, help="Lower value = tighter squeeze requirement.")

@st.cache_data
def get_gmma_data(symbol, start):
    df = yf.download(symbol, start=start)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # GMMA Ribbons
    st_p, lt_p = [3, 5, 8, 10, 12, 15], [30, 35, 40, 45, 50, 60]
    for p in st_p: df[f'ST_{p}'] = ta.ema(df['Close'], length=p)
    for p in lt_p: df[f'LT_{p}'] = ta.ema(df['Close'], length=p)
    
    # Compression Logic
    df['LT_Max'] = df[[f'LT_{p}' for p in lt_p]].max(axis=1)
    df['LT_Min'] = df[[f'LT_{p}' for p in lt_p]].min(axis=1)
    df['LT_Width_Pct'] = ((df['LT_Max'] - df['LT_Min']) / df['LT_Min']) * 100
    
    df['ST_Min'] = df[[f'ST_{p}' for p in st_p]].min(axis=1)
    return df

df = get_gmma_data(ticker, start_date)

if not df.empty:
    # --- LIVE COMPRESSION ALERT ---
    current_width = df['LT_Width_Pct'].iloc[-1]
    is_compressed = current_width <= compression_threshold
    
    st.sidebar.subheader("📢 Real-Time Alerts")
    if is_compressed:
        st.sidebar.warning(f"⚠️ COMPRESSION DETECTED: {ticker} ribbon is squeezed at {current_width:.2f}%. Watch for breakout!")
    else:
        st.sidebar.success(f"✅ Normal Trend: Ribbon width is {current_width:.2f}%")

    # --- BREAKOUT BACKTESTER ---
    trades, in_trade = [], False
    for i in range(1, len(df)):
        st_min, lt_max = df['ST_Min'].iloc[i], df['LT_Max'].iloc[i]
        width, prev_width = df['LT_Width_Pct'].iloc[i], df['LT_Width_Pct'].iloc[i-1]
        
        if not in_trade:
            # Entry: Ribbon Breakout + Expansion Confirmation
            if st_min > lt_max and width > prev_width:
                in_trade = True
                # Check if it came out of a squeeze
                was_squeezed = any(df['LT_Width_Pct'].iloc[i-10:i] <= compression_threshold)
                trades.append({
                    "Date": df.index[i].date(),
                    "Type": "🚀 Squeeze Breakout" if was_squeezed else "📈 Trend Continuation",
                    "Entry P": round(df['Close'].iloc[i], 2)
                })
        elif in_trade:
            if st_min < lt_max: # Ribbon Cross Exit
                in_trade = False
                pnl = ((df['Close'].iloc[i] - trades[-1]['Entry P']) / trades[-1]['Entry P']) * 100
                trades[-1]['Exit Date'] = df.index[i].date()
                trades[-1]['ROI %'] = round(pnl, 2)

    if trades:
        st.write("### 🏆 Squeeze & Breakout History")
        st.table(pd.DataFrame(trades))