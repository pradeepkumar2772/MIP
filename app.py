import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime

# --- Page Configuration ---
st.set_page_config(page_title="RS Comparative Report", layout="wide")
st.title("📊 Relative Strength Comparative Report")

# --- User Inputs ---
ticker = st.sidebar.text_input("Numerator (Stock)", "BRITANNIA.NS")
benchmark = st.sidebar.text_input("Denominator (Benchmark)", "^NSEI")
timeframe = st.sidebar.selectbox("Timeframe", ["65D", "125D", "250D"])
lookback = int(timeframe.replace('D', ''))

@st.cache_data
def get_rs_report_data(stock, bench, days):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days + 50) # Buffer for EMA
    
    s_data = yf.download(stock, start=start, end=end)['Close']
    b_data = yf.download(bench, start=start, end=end)['Close']
    
    if s_data.empty or b_data.empty: return pd.DataFrame()
    
    df = pd.DataFrame(index=s_data.index)
    df['Stock_Price'] = s_data
    df['Bench_Price'] = b_data
    
    # 1. Ratio Calculation
    df['RS_Ratio'] = df['Stock_Price'] / df['Bench_Price']
    
    # 2. RS Trend (EMA of the Ratio)
    df['RS_EMA'] = ta.ema(df['RS_Ratio'], length=20)
    
    # 3. Ratio Performance (% Change in Ratio)
    df['Ratio_Perf'] = df['RS_Ratio'].pct_change(days) * 100
    
    return df.tail(lookback)

df = get_rs_report_data(ticker, benchmark, lookback)

if not df.empty:
    # --- Generating the Report Table ---
    report = []
    
    curr_ratio = df['RS_Ratio'].iloc[-1]
    curr_ema = df['RS_EMA'].iloc[-1]
    perf = df['Ratio_Perf'].iloc[-1]
    
    # Commentary Logic
    if perf > 0:
        comment = f"{ticker} outperformed {benchmark}."
    else:
        comment = f"RS Divergence. {ticker} is underperforming {benchmark}."
        
    # RS Trend Logic
    rs_trend = "Bullish - Continuation" if curr_ratio > curr_ema else "Relative Weakness - Bearish"
    
    report.append({
        "Scrip": ticker,
        "Closing Price": round(df['Stock_Price'].iloc[-1], 2),
        "Benchmark Trend": "Up" if df['Bench_Price'].iloc[-1] > df['Bench_Price'].iloc[-5] else "Down",
        "Commentary": comment,
        "Ratio Performance": f"{round(perf, 2)}%",
        "RS Trend": rs_trend
    })

    st.write(f"### Output Window ({timeframe})")
    st.table(pd.DataFrame(report))