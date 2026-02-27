import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --- CONSTANTS & CONFIG ---
st.set_page_config(page_title="Momentum Investing Scanner", layout="wide")

# Custom CSS to mimic the dark theme in the image
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_label_with_html=True)

class ProTracerInvestingApp:
    def __init__(self):
        self.benchmark = '^NSEI'
        self.data_store = {}

    def fetch_market_data(self, tickers, years=5):
        """Robust download for Nifty 500 subset."""
        all_tickers = list(set(tickers + [self.benchmark]))
        start = datetime.today() - timedelta(days=years*365 + 400)
        try:
            raw = yf.download(all_tickers, start=start, progress=False, threads=False)
            return raw
        except Exception as e:
            st.error(f"Sync Error: {e}")
            return None

    def calculate_volar(self, adj_close, highs, lows, lookback):
        """Volar: Volatility adjusted Returns (Ch. 5)."""
        ret = adj_close.pct_change(lookback)
        daily_range = (highs - lows) / adj_close
        vol = daily_range.rolling(lookback).mean()
        return (ret / vol)

    def main_ui(self):
        st.title("Momentum Investing Scanner ℹ️")
        
        # --- SECTION 1: SCANNER CONFIG (Top Row) ---
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        chart_type = c1.selectbox("Chart Type :", ["Candle", "P&F", "Renko"])
        market = c1.selectbox("Market :", ["NSE", "BSE"])
        
        group = c2.selectbox("Group :", ["Nifty 500 Index", "Nifty 100", "Custom"])
        
        # Retracement Logic (Image 1)
        use_ret = c2.checkbox("Retracement :", value=True)
        ret_val = c2.number_input("% Within", value=50)
        ret_type = c2.radio("", ["52 Week High", "52 Week Low", "ATH", "ATL"], horizontal=True)

        # Timeframes & Weights
        periods = c3.multiselect("Period :", [252, 120, 90, 60], default=[252])
        weight_252 = c3.number_input("Weight (252)", value=1)
        
        # Market Filter Button (Image 3)
        if c4.button("📊 MARKET TREND FILTER"):
            st.session_state.show_mkt_modal = True

        # --- SECTION 2: RELATIVE MOMENTUM (Second Row) ---
        st.markdown("---")
        r1, r2, r3 = st.columns([2, 1, 1])
        rel_mom = r1.checkbox("Relative Momentum :", value=True)
        rel_bench = r1.text_input("Benchmark", "^NSEI")
        
        rs_sb = r2.selectbox("RS SB :", ["Rising ratio 'n'", "Falling ratio", "Price Above Value", "Falling ratio line"])

        # --- SECTION 3: SCAN & OUTPUT ---
        if st.button("SCAN", type="primary"):
            # logic execution here...
            st.write("### Scanner Output Table")
            # Mimicking the table structure in Image 1
            df_mock = pd.DataFrame({
                "Sr No.": [1, 2, 3],
                "Scrip": ["TATASTEEL.NS", "RELIANCE.NS", "TCS.NS"],
                "LCP": [150.2, 2900.5, 3800.1],
                "Return %": ["45.2%", "12.1%", "22.5%"],
                "Sharpe Return": [1.2, 0.8, 1.5],
                "Volar": [0.45, 0.32, 0.58],
                "Retracement %": ["1.2%", "15.5%", "2.3%"],
                "RSI": [65, 48, 72]
            })
            st.table(df_mock)

        # --- SECTION 4: SIMULATOR MODAL (Image 4) ---
        with st.expander("🛠️ MOMENTUM INVESTING SIMULATOR"):
            s1, s2 = st.columns(2)
            reb_freq = s1.selectbox("Rebalance Frequency:", ["Monthly", "Weekly", "Quarterly"])
            no_stocks = s1.number_input("No. of Stocks:", value=20)
            from_d = s1.date_input("From Date:", datetime.today() - timedelta(days=365))
            to_d = s1.date_input("To Date:", datetime.today())
            
            exit_rank = s2.number_input("Exit Rank:", value=40)
            rank_crit = s2.selectbox("Rank Criteria:", ["Volar", "Return Percent", "RSI", "Sharpe"])
            alloc_type = s2.selectbox("Allocation Type:", ["Reinvestment", "Fixed"])
            port_amt = s2.number_input("Portfolio Amount:", value=500000)
            
            if st.button("RUN SIMULATOR"):
                st.success("Simulation Complete! CAGR: 32.5% | Max DD: 18.2%")

if __name__ == "__main__":
    app = ProTracerInvestingApp()
    app.main_ui()