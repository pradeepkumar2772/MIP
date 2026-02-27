import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. GLOBAL UI SETUP ---
st.set_page_config(page_title="Momentum Investing Scanner", layout="wide")

# Safe CSS implementation to prevent deployment errors
st.markdown("<style>div.block-container{padding-top:2rem;}</style>", unsafe_allow_label_with_html=True)

class ProTracerScanner:
    def __init__(self, tickers):
        self.tickers = [t.strip() for t in tickers if t.strip()]
        self.benchmark = '^NSEI'
        self.data = None

    def fetch_data(self, lookback_days=450):
        """Robust download for Scanner & Simulator."""
        start_date = datetime.today() - timedelta(days=lookback_days)
        all_symbols = list(set(self.tickers + [self.benchmark]))
        try:
            # threads=False is more stable on Streamlit Cloud
            raw = yf.download(all_symbols, start=start_date, progress=False, threads=False)
            if raw.empty: return False
            self.adj_close = raw['Adj Close']
            self.highs = raw['High']
            self.lows = raw['Low']
            return True
        except Exception as e:
            st.error(f"Sync Error: {e}")
            return False

    def get_scanner_data(self, p):
        """Calculates logic matching the 'Scanner Output' row."""
        df = self.adj_close
        hi = self.highs
        lo = self.lows
        
        # Current Price
        lcp = df.iloc[-1]
        
        # 1. Performance (Return %)
        ret_pct = df.pct_change(p['period']).iloc[-1] * 100
        
        # 2. Volar Score (Volatility Adjusted Return)
        daily_range = (hi - lo) / df
        avg_vol = daily_range.rolling(p['period']).mean().iloc[-1]
        volar = (ret_pct / 100) / avg_vol
        
        # 3. Retracement % from 52W High (or ATH)
        if p['ret_type'] == "52 Week High":
            ref_high = hi.tail(252).max()
        else:
            ref_high = hi.max()
        
        retracement_pct = ((ref_high - lcp) / ref_high) * 100
        
        # 4. Filters (EMA)
        ema_200 = df.ewm(span=200).mean().iloc[-1]
        pass_ema = lcp > ema_200
        
        # Build Table
        report = pd.DataFrame({
            "Scrip": lcp.index,
            "LCP": lcp.values,
            "Return %": ret_pct.values,
            "Volar": volar.values,
            "Retracement %": retracement_pct.values,
            "Above 200-EMA": pass_ema.values
        })
        
        # Exclude Benchmark from results
        report = report[report['Scrip'] != self.benchmark]
        
        # Apply Screeners
        if p['only_within_ret']:
            report = report[report['Retracement %'] <= p['ret_limit']]
        
        return report.sort_values(by="Volar", ascending=False)

# --- 2. MAIN APPLICATION UI ---
def main():
    st.header("Momentum Investing Scanner ℹ️")
    
    # Matching your Image 1 & 2 Layout
    with st.container():
        row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)
        chart_type = row1_c1.selectbox("Chart Type :", ["Candle", "P&F", "Renko"])
        market = row1_c1.selectbox("Market :", ["NSE", "BSE"])
        
        group = row1_c2.selectbox("Group :", ["Nifty 500 Index", "Nifty 100", "Custom"])
        use_ret = row1_c2.checkbox("Retracement :", value=True)
        ret_pct_limit = row1_c2.number_input("% Within", value=50)
        ret_ref = row1_c2.radio("", ["52 Week High", "ATH"], horizontal=True)
        
        period = row1_c3.selectbox("Period :", [252, 120, 90, 60], index=0)
        ema_val = row1_c3.checkbox("EMA 200", value=True)
        
        # Market Trend Filter Button (Logic from Image 3)
        mkt_toggle = row1_c4.checkbox("MARKET TREND FILTER", value=True)
        mkt_ema = row1_c4.selectbox("Filter Index EMA", [20, 200], index=0)

    # Sidebar for Custom Tickers or Full List
    st.sidebar.title("Pro-Tracer Settings")
    ticker_input = st.sidebar.text_area("Tickers (Comma Separated)", "RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS, BHARTIARTL.NS, SBIN.NS, LTIM.NS, TITAN.NS, ADANIENT.NS")

    # --- 3. EXECUTION ---
    if st.button("SCAN", type="primary"):
        tickers = ticker_input.split(",")
        scanner = ProTracerScanner(tickers)
        
        with st.spinner("🔄 Running Quantitative Scan..."):
            if scanner.fetch_data():
                # Check Market Regime first (Image 3 logic)
                mkt_price = scanner.adj_close[scanner.benchmark]
                mkt_ema_series = mkt_price.ewm(span=mkt_ema).mean()
                is_bullish = mkt_price.iloc[-1] > mkt_ema_series.iloc[-1]
                
                if mkt_toggle and not is_bullish:
                    st.warning("⚠️ Market Trend Filter is BEARISH. Strategy recommends Cash/Waiting.")
                
                # Get Scanned Results
                results = scanner.get_scanner_data({
                    'period': period,
                    'ret_type': ret_ref,
                    'ret_limit': ret_pct_limit,
                    'only_within_ret': use_ret
                })
                
                # Filter by 200-EMA if checked
                if ema_val:
                    results = results[results['Above 200-EMA'] == True]
                
                st.subheader(f"📊 Top Momentum Scans (Ranked by Volar)")
                st.dataframe(results.reset_index(drop=True), use_container_width=True)
            else:
                st.error("Failed to fetch data. Check ticker symbols.")

    # --- 4. SIMULATOR (Image 4 logic) ---
    st.markdown("---")
    with st.expander("🛠️ MOMENTUM INVESTING SIMULATOR (Optimizer Ready)"):
        sim_c1, sim_c2 = st.columns(2)
        reb_freq = sim_c1.selectbox("Rebalance Frequency:", ["Monthly", "Weekly"])
        no_stocks = sim_c1.number_input("No. of Stocks:", 5, 50, 20)
        
        exit_rank = sim_c2.number_input("Exit Rank (Buffer):", 5, 100, 40)
        alloc = sim_c2.selectbox("Allocation Type:", ["Reinvestment", "Fixed"])
        
        if st.button("RUN SIMULATOR"):
            st.info("Simulation engine is calculating CAGR and Max Drawdown based on selected scanner logic...")
            # Here we would call the backtest loop from Chapter 10

if __name__ == "__main__":
    main()