import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime

st.set_page_config(page_title="NSE RS Scanner", layout="wide")
st.title("📊 Sectoral Relative Strength Report")

# --- Timeframe & Denominator Settings ---
bench_ticker = "^NSEI" # Nifty 50 is the default denominator
intervals = {"Hourly": "1h", "Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
selected_int = st.sidebar.selectbox("Select Timeframe", list(intervals.keys()))
lookback_map = {"Hourly": 7, "Daily": 365, "Weekly": 730, "Monthly": 1825}

# Sector List from your uploaded image
sector_dict = {
    "Nifty Auto": "^CNXAUTO", "Nifty Bank": "^NSEBANK", "Nifty IT": "^CNXIT",
    "Nifty FMCG": "^CNXFMCG", "Nifty Pharma": "^CNXPHARMA", "Nifty Metal": "^CNXMETAL",
    "Nifty Reality": "^CNXREALTY", "Nifty Energy": "^CNXENERGY", "Nifty Infra": "^CNXINFRA",
    "Nifty PSE": "^CNXPSE", "Nifty CPSE": "NIFTY_CPSE.NS", "Nifty MNC": "^CNXMNC"
}

@st.cache_data
def get_report(sectors, benchmark, interval, days):
    report_data = []
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    
    # Get Benchmark
    b_df = yf.download(benchmark, start=start, interval=intervals[interval])['Close']
    if b_df.empty: return pd.DataFrame()
    b_trend = "Up" if b_df.iloc[-1] > b_df.iloc[-5] else "Down"
    
    for name, ticker in sectors.items():
        s_df = yf.download(ticker, start=start, interval=intervals[interval])['Close']
        if s_df.empty: continue
        
        # 1. Ratio Performance Calculation
        ratio = s_df / b_df
        ratio_perf = ((ratio.iloc[-1] - ratio.iloc[-20]) / ratio.iloc[-20]) * 100
        
        # 2. RS Trend (EMA of Ratio)
        rs_ema = ratio.ewm(span=20).mean()
        rs_trend = "Bullish - Continuation" if ratio.iloc[-1] > rs_ema.iloc[-1] else "Relative Weakness - Bearish"
        
        # 3. Commentary Logic
        if ratio_perf > 0:
            comment = f"{name} went up with {benchmark} and outperformed."
        else:
            comment = f"RS Divergence. {name} is underperforming {benchmark}."
            
        report_data.append({
            "Scrip Report": name,
            "Closing Price": round(s_df.iloc[-1], 2),
            "Benchmark Trend": b_trend,
            "Commentary": comment,
            "Ratio Performance": f"{round(ratio_perf, 2)}%",
            "RS Trend": rs_trend
        })
    return pd.DataFrame(report_data)

report_df = get_report(sector_dict, bench_ticker, selected_int, lookback_map[selected_int])

if not report_df.empty:
    st.write(f"### Output Window (Denominator: Nifty 50 | Timeframe: {selected_int})")
    st.table(report_df)