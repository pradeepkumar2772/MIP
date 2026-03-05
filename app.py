import yfinance as yf
import pandas as pd
import streamlit as st
import datetime

st.set_page_config(page_title="NSE Sector Scanner", layout="wide")
st.title("📊 NSE Sector Outperformance Scanner")

# Define NSE Sectoral Indices
sectors = {
    "Nifty 50": "^NSEI",
    "Nifty Bank": "^NSEBANK",
    "Nifty IT": "^CNXIT",
    "Nifty FMCG": "^CNXFMCG",
    "Nifty Pharma": "^CNXPHARMA",
    "Nifty Metal": "^CNXMETAL",
    "Nifty Auto": "^CNXAUTO",
    "Nifty Realty": "^CNXREALTY",
    "Nifty Energy": "^CNXENERGY"
}

start_date = st.sidebar.date_input("Start Date", datetime.date(2026, 1, 1))

@st.cache_data
def get_sector_performance(sector_dict, start):
    performance_data = []
    
    # Get Nifty 50 Benchmark
    nifty = yf.download(sector_dict["Nifty 50"], start=start)
    if nifty.empty: return pd.DataFrame()
    if isinstance(nifty.columns, pd.MultiIndex): nifty.columns = nifty.columns.get_level_values(0)
    
    # Calculate Nifty % Return
    nifty_start_p = float(nifty['Close'].iloc[0])
    nifty_end_p = float(nifty['Close'].iloc[-1])
    nifty_perf = ((nifty_end_p - nifty_start_p) / nifty_start_p) * 100
    
    for name, ticker in sector_dict.items():
        if name == "Nifty 50": continue
        
        data = yf.download(ticker, start=start)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            
            # Absolute Return Calculation
            s_start_p = float(data['Close'].iloc[0])
            s_end_p = float(data['Close'].iloc[-1])
            abs_return = ((s_end_p - s_start_p) / s_start_p) * 100
            
            # FIXED: relative_strength is now a scalar (single number)
            relative_strength = abs_return - nifty_perf
            
            performance_data.append({
                "Sector": name,
                "Absolute Return %": round(abs_return, 2),
                "Relative vs Nifty %": round(relative_strength, 2),
                "Status": "✅ Outperforming" if relative_strength > 0 else "❌ Underperforming"
            })
            
    return pd.DataFrame(performance_data).sort_values("Relative vs Nifty %", ascending=False)

perf_df = get_sector_performance(sectors, start_date)

if not perf_df.empty:
    st.write(f"### Sector Performance (YTD 2026)")
    st.table(perf_df)
    
    # Leaderboard Logic
    leader = perf_df.iloc[0]
    st.success(f"🏆 **Current Sector Leader:** {leader['Sector']} (+{leader['Relative vs Nifty %']}% Alpha)")