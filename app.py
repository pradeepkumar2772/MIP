import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v8.0", layout="wide")

st.title("🛡️ Pro-Tracer: Pure GMMA Trend Engine")
st.sidebar.header("Strategy Settings")

ticker = st.sidebar.text_input("Ticker Symbol", "BRITANNIA.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# --- GMMA Parameter Inputs ---
st.sidebar.subheader("Expansion Filters")
tsl_pct = st.sidebar.slider("Trailing Stop %", 5.0, 20.0, 10.0)
min_expansion = st.sidebar.slider("Min Ribbon Width %", 0.1, 5.0, 0.5, help="Filters out narrow/flat ribbons.")

@st.cache_data
def get_gmma_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Short-Term Group (3, 5, 8, 10, 12, 15)
    st_emas = [3, 5, 8, 10, 12, 15]
    for e in st_emas:
        df[f'EMA_{e}'] = ta.ema(df['Close'], length=e)
    df['ST_Max'] = df[[f'EMA_{e}' for e in st_emas]].max(axis=1)
    df['ST_Min'] = df[[f'EMA_{e}' for e in st_emas]].min(axis=1)
    
    # Long-Term Group (30, 35, 40, 45, 50, 60)
    lt_emas = [30, 35, 40, 45, 50, 60]
    for e in lt_emas:
        df[f'EMA_{e}'] = ta.ema(df['Close'], length=e)
    df['LT_Max'] = df[[f'EMA_{e}' for e in lt_emas]].max(axis=1)
    df['LT_Min'] = df[[f'EMA_{e}' for e in lt_emas]].min(axis=1)
    
    # Calculate Expansion Widths
    df['LT_Width_Pct'] = ((df['LT_Max'] - df['LT_Min']) / df['LT_Min']) * 100
    return df

df = get_gmma_data(ticker, start_date, end_date)

if not df.empty:
    st.header(f"📈 GMMA Trend Analysis: {ticker}")
    
    trades, in_trade, peak_p = [], False, 0
    
    for i in range(1, len(df)):
        curr_p = float(df['Close'].iloc[i])
        st_min = df['ST_Min'].iloc[i]
        lt_max = df['LT_Max'].iloc[i]
        lt_width = df['LT_Width_Pct'].iloc[i]
        prev_lt_width = df['LT_Width_Pct'].iloc[i-1]
        
        if not in_trade:
            # ENTRY LOGIC:
            # 1. Short-term ribbon is entirely above Long-term ribbon
            # 2. Long-term ribbon is expanding (Institutional conviction)
            if st_min > lt_max and lt_width > prev_lt_width and lt_width > min_expansion:
                in_trade, entry_date, entry_price = True, df.index[i], curr_p
                peak_p = curr_p
        
        elif in_trade:
            peak_p = max(peak_p, curr_p)
            tsl_price = peak_p * (1 - tsl_pct / 100)
            
            # EXIT LOGIC:
            # 1. Trailing Stop Hit
            # 2. Short-term ribbon crosses back into Long-term (Trend Break)
            if curr_p <= tsl_price or st_min < lt_max:
                in_trade = False
                pnl = ((curr_p - entry_price) / entry_price) * 100
                trades.append({
                    "Entry": entry_date.date(),
                    "Exit": df.index[i].date(),
                    "ROI %": round(pnl, 2),
                    "Exit Reason": "TSL" if curr_p <= tsl_price else "Ribbon Break"
                })

    if trades:
        t_df = pd.DataFrame(trades)
        st.subheader("🏆 Strategy Performance")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total ROI", f"{t_df['ROI %'].sum():.1f}%")
        c2.metric("Win Rate", f"{(len(t_df[t_df['ROI %'] > 0])/len(t_df))*100:.1f}%")
        c3.metric("Avg Trade", f"{t_df['ROI %'].mean():.2f}%")
        
        st.dataframe(t_df, use_container_width=True)
    else:
        st.warning("No GMMA expansion trades found with current filters.")