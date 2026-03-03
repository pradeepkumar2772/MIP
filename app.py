import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer Scanner", layout="wide")

st.title("📡 Pro-Tracer: Nifty 50 Multi-Stock Screener")

# --- Nifty 50 Ticker List ---
NIFTY50_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS",
    "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LTIM.NS",
    "LT.NS", "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS",
    "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS"
]

st.sidebar.header("Scanner Settings")
rsi_len = st.sidebar.number_input("RSI Length", value=14)
ema_len = st.sidebar.number_input("Signal EMA Length", value=9)
trend_ema = st.sidebar.number_input("Trend Filter (EMA 200)", value=200)

if st.button("🔍 Scan Nifty 50 Now"):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. Batch Download (Faster than individual downloads)
    status_text.text("Downloading Nifty 50 Data...")
    data = yf.download(NIFTY50_TICKERS, period="1y", group_by='ticker', progress=False)
    
    for i, ticker in enumerate(NIFTY50_TICKERS):
        try:
            df = data[ticker].copy()
            df.dropna(inplace=True)
            
            # 2. Calculate Indicators
            df['RSI'] = ta.rsi(df['Close'], length=rsi_len)
            df['RSI_EMA'] = ta.ema(df['RSI'], length=ema_len)
            df['Trend'] = ta.ema(df['Close'], length=trend_ema)
            
            # 3. Get Latest Values
            last_price = df['Close'].iloc[-1]
            last_rsi = df['RSI'].iloc[-1]
            last_ema = df['RSI_EMA'].iloc[-1]
            last_trend = df['Trend'].iloc[-1]
            
            # 4. Define Signal Status
            # Momentum: RSI > Signal EMA
            # Trend: Price > Trend EMA
            momentum_status = "🟢 Bullish" if last_rsi > last_ema else "🔴 Bearish"
            trend_status = "🟢 Above Trend" if last_price > last_trend else "🔴 Below Trend"
            
            action = "🚀 BUY SIGNAL" if (last_rsi > last_ema and last_price > last_trend) else "⏳ WAIT"
            
            results.append({
                "Ticker": ticker,
                "LTP": round(last_price, 2),
                "RSI": round(last_rsi, 2),
                "Momentum": momentum_status,
                "Trend": trend_status,
                "Action": action
            })
            
        except Exception as e:
            continue
            
        progress_bar.progress((i + 1) / len(NIFTY50_TICKERS))
    
    # 5. Display Results
    status_text.text("Scan Complete!")
    scan_df = pd.DataFrame(results)
    
    # Highlighting the "BUY" signals
    st.subheader("📊 Live Market Scanner Results")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Stocks", len(NIFTY50_TICKERS))
    col2.metric("Buy Signals", len(scan_df[scan_df['Action'] == "🚀 BUY SIGNAL"]))
    col3.metric("Bearish Momentum", len(scan_df[scan_df['Momentum'] == "🔴 Bearish"]))

    # Show only Buy Signals first
    st.write("### 🚀 Active Buy Signals")
    st.table(scan_df[scan_df['Action'] == "🚀 BUY SIGNAL"])
    
    st.write("### 📋 All Stocks Overview")
    st.dataframe(scan_df, use_container_width=True)