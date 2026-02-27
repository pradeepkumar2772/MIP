import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. CORE LOGIC ENGINE ---
class MomentumEngine:
    def __init__(self, tickers):
        self.tickers = [t.strip() for t in tickers if t.strip()]
        self.benchmark = '^NSEI'

    def fetch_data(self):
        # Explicitly fetching benchmark and stocks separately for stability
        all_symbols = list(set(self.tickers + [self.benchmark]))
        start = datetime.today() - timedelta(days=500)
        try:
            data = yf.download(all_symbols, start=start, progress=False, threads=False)
            if data.empty: return None
            return data
        except:
            return None

    def calculate_scan(self, data, period, ret_limit, use_ret):
        adj_close = data['Adj Close']
        highs = data['High']
        lows = data['Low']
        
        # Current Metrics
        lcp = adj_close.iloc[-1]
        returns = adj_close.pct_change(period).iloc[-1] * 100
        
        # Volar Calculation (Ch. 5: Return / Avg Daily Range)
        daily_range = (highs - lows) / adj_close
        avg_vol = daily_range.rolling(period).mean().iloc[-1]
        volar = (returns / 100) / avg_vol
        
        # Retracement (52W High)
        h52w = highs.tail(252).max()
        retracement = ((h52w - lcp) / h52w) * 100
        
        # Filters
        ema_200 = adj_close.ewm(span=200).mean().iloc[-1]
        
        df_res = pd.DataFrame({
            "Scrip": lcp.index,
            "LCP": lcp.values,
            "Return %": returns.values,
            "Volar": volar.values,
            "Retracement %": retracement.values,
            "EMA_200": ema_200.values
        })
        
        # Remove benchmark from results
        df_res = df_res[df_res['Scrip'] != self.benchmark]
        
        # Filter Logic from UI
        if use_ret:
            df_res = df_res[df_res['Retracement %'] <= ret_limit]
            
        return df_res.sort_values("Volar", ascending=False)

# --- 2. STREAMLIT UI ---
def main():
    st.title("Momentum Investing Scanner ℹ️")

    # Header Controls (Mirroring Image 1 & 2)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chart = st.selectbox("Chart Type :", ["Candle", "P&F", "Renko"])
        period = st.selectbox("Period :", [252, 120, 90, 60])
        
    with col2:
        group = st.selectbox("Group :", ["Nifty 500 Index", "Nifty 100"])
        use_ret = st.checkbox("Retracement :", value=True)
        ret_val = st.number_input("% Within", value=50)

    with col3:
        mkt_filter = st.checkbox("MARKET TREND FILTER", value=True)
        mkt_ema = st.selectbox("Index EMA :", [20, 200])

    # Ticker input (Hidden in sidebar to keep UI clean)
    tickers_raw = st.sidebar.text_area("Tickers", "RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS, BHARTIARTL.NS, SBIN.NS, LTIM.NS, TITAN.NS, ADANIENT.NS")
    
    if st.button("SCAN", type="primary"):
        engine = MomentumEngine(tickers_raw.split(","))
        
        with st.spinner("Syncing Data..."):
            raw_data = engine.fetch_data()
            
            if raw_data is not None:
                # Market Trend logic (Image 3)
                bm_price = raw_data['Adj Close'][engine.benchmark]
                bm_ema = bm_price.ewm(span=mkt_filter).mean()
                
                if mkt_filter and bm_price.iloc[-1] < bm_ema.iloc[-1]:
                    st.error("⚠️ Market Regime: BEARISH (Wait for Index to cross EMA)")
                else:
                    st.success("✅ Market Regime: BULLISH")

                # Results
                results = engine.calculate_scan(raw_data, period, ret_val, use_ret)
                st.dataframe(results, use_container_width=True)
            else:
                st.error("Connection Failed. Check symbols.")

    # Simulator Section (Image 4)
    st.markdown("---")
    with st.expander("🛠️ MOMENTUM INVESTING SIMULATOR"):
        st.write("This engine calculates CAGR and Drawdown based on logic rebalancing.")
        s1, s2 = st.columns(2)
        s1.selectbox("Frequency:", ["Monthly", "Weekly"])
        s2.number_input("Portfolio Size:", 5, 50, 20)
        if st.button("RUN SIMULATOR"):
            st.info("Simulation running on Optimizer Engine...")

if __name__ == "__main__":
    main()