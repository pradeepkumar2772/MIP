import yfinance as yf
import pandas as pd
import streamlit as st
import datetime

st.set_page_config(page_title="NSE Mega RS Scanner", layout="wide")
st.title("📊 Complete Sectoral Relative Strength Report")

# --- 1. FULL SECTOR LIST (Corrected Tickers) ---
sectors = {
    "NIFTY CONSR DURBL": "NIFTY_CONSR_DURBL.NS", "NIFTY HEALTHCARE": "NIFTY_HEALTHCARE.NS",
    "NIFTY IND DIGITAL": "NIFTY_IND_DIGITAL.NS", "NIFTY INDIA MFG": "NIFTY_INDIA_MFG.NS",
    "NIFTY OIL AND GAS": "NIFTY_OIL_AND_GAS.NS", "Nifty 50": "^NSEI",
    "Nifty Auto": "^CNXAUTO", "Nifty Bank": "^NSEBANK", "Nifty CPSE": "NIFTY_CPSE.NS",
    "Nifty Capital Mkt": "NIFTY_CAP_MKT.NS", "Nifty Chemicals": "NIFTY_CHEMICALS.NS",
    "Nifty Commodities": "^CNXCMDT", "Nifty Consumption": "^CNXCONSUMP",
    "Nifty CoreHousing": "NIFTY_CORE_HOUSING.NS", "Nifty Corp MAATR": "NIFTY_CORP_MAATR.NS",
    "Nifty EV": "NIFTY_EV_NEW_AGE.NS", "Nifty Energy": "^CNXENERGY", "Nifty FMCG": "^CNXFMCG",
    "Nifty Fin Service": "^CNXFIN", "Nifty FinSerExBnk": "NIFTY_FIN_SRV_EX_BANK.NS",
    "Nifty FinSrv25 50": "NIFTY_FIN_SRV_25_50.NS", "Nifty Housing": "NIFTY_HOUSING.NS",
    "Nifty IT": "^CNXIT", "Nifty Ind Defence": "NIFTY_IND_DEFENCE.NS",
    "Nifty Ind Tourism": "NIFTY_IND_TOURISM.NS", "Nifty Infra": "^CNXINFRA",
    "Nifty InfraLog": "NIFTY_INFRA_LOG.NS", "Nifty Internet": "NIFTY_INTERNET.NS",
    "Nifty MNC": "^CNXMNC", "Nifty MS Fin Serv": "NIFTY_MS_FIN_SRV.NS",
    "Nifty MS IT Telcm": "NIFTY_MS_IT_TELCM.NS", "Nifty MS Ind Cons": "NIFTY_MS_IND_CONS.NS",
    "Nifty Media": "^CNXMEDIA", "Nifty Metal": "^CNXMETAL", "Nifty MidSml Hlth": "NIFTY_MIDSML_HLTH.NS",
    "Nifty Mobility": "NIFTY_MOBILITY.NS", "Nifty New Consump": "NIFTY_NEW_CONSUMP.NS",
    "Nifty NonCyc Cons": "NIFTY_NON_CYC_CONS.NS", "Nifty PSE": "^CNXPSE",
    "Nifty PSU Bank": "^CNXPSUBANK", "Nifty Pharma": "^CNXPHARMA", "Nifty Pvt Bank": "NIFTY_PVT_BANK.NS",
    "Nifty Realty": "^CNXREALTY", "Nifty Rural": "NIFTY_RURAL.NS",
    "Nifty Serv Sector": "^CNXSERVICE", "Nifty Trans Logis": "NIFTY_TRANS_LOGIS.NS",
    "Nifty Waves": "NIFTY_WAVES.NS"
}

# --- 2. TIMEFRAME SELECTION ---
intervals = {"Hourly": "1h", "Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
selected_int = st.sidebar.selectbox("Select Timeframe", list(intervals.keys()))

@st.cache_data
def get_mega_report(sector_map, interval_name):
    report_list = []
    interval = intervals[interval_name]
    
    # Calculate Data Buffer
    days_back = 50 if interval_name == "Hourly" else 730
    start_date = datetime.datetime.now() - datetime.timedelta(days=days_back)
    
    # Benchmark Data
    b_data = yf.download("^NSEI", start=start_date, interval=interval)['Close'].squeeze()
    if b_data.empty: return pd.DataFrame()
    
    b_curr, b_prev = float(b_data.iloc[-1]), float(b_data.iloc[-5])
    b_trend = "Up" if b_curr > b_prev else "Down"
    
    for name, ticker in sector_map.items():
        if name == "Nifty 50": continue
        try:
            s_data = yf.download(ticker, start=start_date, interval=interval)['Close'].squeeze()
            if s_data.empty: continue
            
            # Ratio & Performance
            ratio = s_data / b_data
            ratio_end = float(ratio.iloc[-1])
            ratio_start = float(ratio.iloc[-20]) if len(ratio) > 20 else float(ratio.iloc[0])
            perf = ((ratio_end - ratio_start) / ratio_start) * 100
            
            # RS Trend (EMA Cross)
            rs_ema = ratio.ewm(span=20).mean().iloc[-1]
            rs_trend = "Bullish - Continuation" if ratio_end > rs_ema else "Relative Weakness - Bearish"
            
            # Commentary Logic based on CSV
            if perf > 0:
                comment = f"{name} went up along with Nifty 50 and outperformed"
            else:
                comment = f"RS Divergence. {name} is underperforming Nifty 50"
                
            report_list.append({
                "Scrip": name,
                "LCP": round(float(s_data.iloc[-1]), 2),
                "Benchmark Trend": b_trend,
                "Commentary": comment,
                "Ratio Performance": f"{round(perf, 2)}%",
                "RS Trend": rs_trend
            })
        except: continue
    return pd.DataFrame(report_list)

final_df = get_mega_report(sectors, selected_int)

if not final_df.empty:
    st.write(f"### Output Window ({selected_int})")
    st.table(final_df)