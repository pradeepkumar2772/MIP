import yfinance as yf
import pandas as pd
import streamlit as st
import datetime

st.set_page_config(page_title="NSE Mega Sector Scanner", layout="wide")
st.title("📊 NSE Mega Sector & Theme Scanner")

# Comprehensive Ticker List based on your uploaded image
sectors = {
    "Nifty 50": "^NSEI",
    "Nifty Consumer Durables": "NIFTY_CONSR_DURBL.NS", 
    "Nifty Healthcare": "NIFTY_HEALTHCARE.NS",
    "Nifty India Digital": "NIFTY_IND_DIGITAL.NS",
    "Nifty India Manufacturing": "NIFTY_INDIA_MFG.NS",
    "Nifty Oil and Gas": "NIFTY_OIL_AND_GAS.NS",
    "Nifty Auto": "^CNXAUTO",
    "Nifty Bank": "^NSEBANK",
    "Nifty CPSE": "NIFTY_CPSE.NS",
    "Nifty Capital Markets": "NIFTY_CAP_MKT.NS",
    "Nifty Chemicals": "NIFTY_CHEMICALS.NS",
    "Nifty Commodities": "^CNXCMDT",
    "Nifty Consumption": "^CNXCONSUMP",
    "Nifty Core Housing": "NIFTY_CORE_HOUSING.NS",
    "Nifty EV & New Age Auto": "NIFTY_EV_NEW_AGE.NS",
    "Nifty Energy": "^CNXENERGY",
    "Nifty FMCG": "^CNXFMCG",
    "Nifty Fin Service": "^CNXFIN",
    "Nifty Housing": "NIFTY_HOUSING.NS",
    "Nifty IT": "^CNXIT",
    "Nifty India Defence": "NIFTY_IND_DEFENCE.NS",
    "Nifty Infrastructure": "^CNXINFRA",
    "Nifty MNC": "^CNXMNC",
    "Nifty Metal": "^CNXMETAL",
    "Nifty Pharma": "^CNXPHARMA",
    "Nifty PSE": "^CNXPSE",
    "Nifty PSU Bank": "^CNXPSUBANK",
    "Nifty Realty": "^CNXREALTY",
    "Nifty Serv Sector": "^CNXSERVICE"
}

start_date = st.sidebar.date_input("Analysis Start Date", datetime.date(2026, 1, 1))

@st.cache_data
def get_mega_performance(sector_dict, start):
    performance_data = []
    
    # Get Nifty 50 Benchmark
    nifty = yf.download(sector_dict["Nifty 50"], start=start)
    if nifty.empty: return pd.DataFrame()
    if isinstance(nifty.columns, pd.MultiIndex): nifty.columns = nifty.columns.get_level_values(0)
    
    n_start, n_end = float(nifty['Close'].iloc[0]), float(nifty['Close'].iloc[-1])
    nifty_perf = ((n_end - n_start) / n_start) * 100
    
    for name, ticker in sector_dict.items():
        if name == "Nifty 50": continue
        try:
            data = yf.download(ticker, start=start)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                
                s_start, s_end = float(data['Close'].iloc[0]), float(data['Close'].iloc[-1])
                abs_return = ((s_end - s_start) / s_start) * 100
                alpha = abs_return - nifty_perf
                
                performance_data.append({
                    "Sector/Theme": name,
                    "Return %": round(abs_return, 2),
                    "Alpha vs Nifty": round(alpha, 2),
                    "Status": "🚀 Leading" if alpha > 0 else "⚓ Lagging"
                })
        except: continue
            
    return pd.DataFrame(performance_data).sort_values("Alpha vs Nifty", ascending=False)

perf_df = get_mega_performance(sectors, start_date)

if not perf_df.empty:
    st.write(f"### Performance Leaderboard since {start_date}")
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    top_theme = perf_df.iloc[0]
    st.info(f"💡 **Trading Tip:** The strongest current theme is **{top_theme['Sector/Theme']}**. Focus your GMMA breakout searches here for higher probability setups.")