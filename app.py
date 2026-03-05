import yfinance as yf
import pandas as pd
import streamlit as st
import datetime

st.set_page_config(page_title="NSE Sector Scanner", layout="wide")
st.title("📊 NSE Sector Outperformance Scanner (vs Nifty 50)")

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
    "Nifty Energy": "^CNXENERGY",
    "Nifty Infra": "^CNXINFRA"
}

start_date = st.sidebar.date_input("Start Date", datetime.date(2025, 1, 1))

@st.cache_data
def get_sector_performance(sector_dict, start):
    performance_data = []
    # Benchmark Data (Nifty 50)
    nifty = yf.download(sector_dict["Nifty 50"], start=start)['Close']
    nifty_perf = ((nifty.iloc[-1] - nifty.iloc[0]) / nifty.iloc[0]) * 100
    
    for name, ticker in sector_dict.items():
        if name == "Nifty 50": continue
        
        data = yf.download(ticker, start=start)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Calculate Absolute Return
            abs_return = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
            # Calculate Relative Strength (Alpha)
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
    st.write(f"### Sector Performance since {start_date}")
    st.table(perf_df)
    
    # Highlight the Top Leader
    leader = perf_df.iloc[0]
    st.success(f"🏆 **Current Sector Leader:** {leader['Sector']} with {leader['Relative vs Nifty %']}% Alpha over Nifty 50.")