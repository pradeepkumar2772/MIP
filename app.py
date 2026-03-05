import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime

st.set_page_config(page_title="NSE Sector RS Report", layout="wide")
st.title("📊 Sectoral Relative Strength Report")

# --- 1. Full Sector List from your provided image ---
sector_dict = {
    "Nifty 50": "^NSEI", "Nifty Bank": "^NSEBANK", "Nifty IT": "^CNXIT",
    "Nifty FMCG": "^CNXFMCG", "Nifty Pharma": "^CNXPHARMA", "Nifty Metal": "^CNXMETAL",
    "Nifty Auto": "^CNXAUTO", "Nifty Realty": "^CNXREALTY", "Nifty Energy": "^CNXENERGY",
    "Nifty Infra": "^CNXINFRA", "Nifty MNC": "^CNXMNC", "Nifty PSE": "^CNXPSE",
    "Nifty PSU Bank": "^CNXPSUBANK", "Nifty PVT Bank": "NIFTY_PVT_BANK.NS",
    "Nifty Consumer Durables": "NIFTY_CONSR_DURBL.NS", "Nifty Healthcare": "NIFTY_HEALTHCARE.NS",
    "Nifty India Digital": "NIFTY_IND_DIGITAL.NS", "Nifty India MFG": "NIFTY_INDIA_MFG.NS",
    "Nifty Oil and Gas": "NIFTY_OIL_AND_GAS.NS", "Nifty EV": "NIFTY_EV_NEW_AGE.NS"
}

# --- 2. Timeframe Selection ---
intervals = {"Hourly": "1h", "Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
selected_int = st.sidebar.selectbox("Select Timeframe", list(intervals.keys()))

@st.cache_data
def get_rs_report(sectors, interval_key):
    report_data = []
    interval = intervals[interval_key]
    
    # Define Lookback based on timeframe
    days_back = 59 if interval_key == "Hourly" else 365
    end_dt = datetime.datetime.now()
    start_dt = end_dt - datetime.timedelta(days=days_back)
    
    # Get Benchmark (Nifty 50)
    b_raw = yf.download("^NSEI", start=start_dt, interval=interval)
    if b_raw.empty: return pd.DataFrame()
    b_df = b_raw['Close'].squeeze() # Ensure it's a single Series
    
    # Fix: Ensure we compare single float values
    b_curr, b_prev = float(b_df.iloc[-1]), float(b_df.iloc[-5])
    b_trend = "Up" if b_curr > b_prev else "Down"
    
    for name, ticker in sectors.items():
        if name == "Nifty 50": continue
        s_raw = yf.download(ticker, start=start_dt, interval=interval)
        if s_raw.empty: continue
        s_df = s_raw['Close'].squeeze()
        
        # Ratio Performance Calculation
        ratio = s_df / b_df
        # Calculate % change in ratio over last 20 periods
        ratio_start = float(ratio.iloc[-20]) if len(ratio) > 20 else float(ratio.iloc[0])
        ratio_end = float(ratio.iloc[-1])
        perf = ((ratio_end - ratio_start) / ratio_start) * 100
        
        # RS Trend (Ratio vs its 20-period EMA)
        rs_ema = ratio.ewm(span=20).mean().iloc[-1]
        trend_status = "Bullish - Continuation" if ratio_end > rs_ema else "Relative Weakness - Bearish"
        
        # Commentary Logic based on your CSV
        if perf > 0:
            commentary = f"{name} outperformed Nifty 50."
        else:
            commentary = f"RS Divergence. {name} is underperforming Nifty 50."
            
        report_data.append({
            "Scrip Report": name,
            "Closing Price": round(float(s_df.iloc[-1]), 2),
            "Benchmark Trend": b_trend,
            "Commentary": commentary,
            "Ratio Performance": f"{round(perf, 2)}%",
            "RS Trend": trend_status
        })
    return pd.DataFrame(report_data)

# Execution
final_df = get_rs_report(sector_dict, selected_int)

if not final_df.empty:
    st.write(f"### NSE Sector Performance Report ({selected_int})")
    st.table(final_df)