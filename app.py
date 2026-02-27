import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Pro-Tracer Nifty 500", layout="wide")

class ProTracerNifty500:
    def __init__(self, tickers):
        # Clean and ensure tickers end with .NS
        self.tickers = [t.strip() if t.endswith('.NS') else f"{t.strip()}.NS" for t in tickers]
        self.benchmark_ticker = '^NSEI'
        self.adj_close = None
        self.highs = None
        self.lows = None
        self.benchmark_data = None

    def fetch_data(self, years=5):
        start_date = datetime.today() - timedelta(days=years*365 + 400)
        
        with st.status("📥 Accessing Market Data...", expanded=True) as status:
            # 1. Download Benchmark separately to ensure it exists
            st.write("Fetching Nifty 50 Benchmark...")
            bm_raw = yf.download(self.benchmark_ticker, start=start_date, progress=False, threads=False)
            if bm_raw.empty:
                st.error("Could not fetch Benchmark (^NSEI). Yahoo Finance might be down.")
                return False
            self.benchmark_data = bm_raw['Adj Close']

            # 2. Download Stock Universe
            st.write(f"Fetching {len(self.tickers)} stocks from Universe...")
            stock_raw = yf.download(self.tickers, start=start_date, progress=False, threads=False)
            
            if stock_raw.empty:
                st.error("Universe download failed.")
                return False

            # 3. Align and Structure
            # Handle single ticker vs multi ticker return types
            if len(self.tickers) == 1:
                self.adj_close = stock_raw['Adj Close'].to_frame()
                self.highs = stock_raw['High'].to_frame()
                self.lows = stock_raw['Low'].to_frame()
            else:
                self.adj_close = stock_raw['Adj Close']
                self.highs = stock_raw['High']
                self.lows = stock_raw['Low']
            
            status.update(label="✅ Data Synced Successfully!", state="complete")
            return True

    def run_engine(self, p):
        # Math references
        df = self.adj_close
        hi = self.highs
        lo = self.lows
        bm = self.benchmark_data
        
        # 1. ALPHA & VOLAR CALCULATIONS
        lookback = p['lookback']
        ret_252 = df.pct_change(lookback)
        daily_range = (hi - lo) / df
        volar = ret_252 / daily_range.rolling(lookback).mean()
        
        # Absolute Trend
        ema_200 = df.ewm(span=200, adjust=False).mean()
        
        # Relative Trend (Stock / Nifty)
        # We reindex benchmark to match stock dates in case of mismatches
        bm = bm.reindex(df.index).ffill()
        ratio = df.div(bm, axis=0)
        ratio_ema = ratio.ewm(span=200, adjust=False).mean()
        
        # Retracement
        h52w = hi.rolling(lookback).max()
        pct_from_high = (h52w - df) / h52w

        # Market Regime
        mkt_ema = bm.ewm(span=p['mkt_filter_ema']).mean()
        mkt_bullish = bm > mkt_ema

        # 2. SIMULATION
        history = []
        portfolio = []
        # Monthly rebalance dates
        valid_dates = df.index[lookback + 20:]
        rebalance_dates = valid_dates[::21] 

        for date in rebalance_dates:
            is_bullish = mkt_bullish.loc[date]
            
            # Blueprint Logic
            eligibility = (
                (df.loc[date] > ema_200.loc[date]) & 
                (ratio.loc[date] > ratio_ema.loc[date]) & 
                (pct_from_high.loc[date] <= p['retracement'])
            )
            
            eligible_stocks = eligibility[eligibility].index.tolist()
            
            # Ranking by Volar
            if not eligible_stocks:
                scores = pd.Series(dtype=float)
            else:
                scores = volar.loc[date, eligible_stocks].sort_values(ascending=False)

            # Rebalance with Buffer
            # Keep existing if still in top (Size + Buffer)
            portfolio = [s for s in portfolio if s in scores.index and 
                         scores.index.get_loc(s) < (p['size'] + p['buffer'])]
            
            # Add new entries if Market is Bullish
            if is_bullish:
                needed = p['size'] - len(portfolio)
                new_adds = [s for s in scores.index if s not in portfolio][:needed]
                portfolio.extend(new_adds)
            
            history.append({
                'Date': date.date(),
                'Mkt_Regime': "BULLISH" if is_bullish else "CAUTION",
                'Count': len(portfolio),
                'Basket': ", ".join(portfolio)
            })

        return pd.DataFrame(history)

# --- UI ---
def main():
    st.title("🚀 Pro-Tracer: Nifty 500 Strategy Engine")
    
    # Nifty 500 Sample (Add your full list here)
    nifty_500_sample = "ABB.NS, ADANIENT.NS, ADANIPORTS.NS, AMBUJACEM.NS, APOLLOHOSP.NS, ASIANPAINT.NS, AXISBANK.NS, BAJAJ-AUTO.NS, BAJAJFINSV.NS, BAJFINANCE.NS, BHARTIARTL.NS, BPCL.NS, BRITANNIA.NS, CIPLA.NS, COALINDIA.NS, DIVISLAB.NS, DRREDDY.NS, EICHERMOT.NS, GRASIM.NS, HCLTECH.NS, HDFCBANK.NS, HDFCLIFE.NS, HEROMOTOCO.NS, HINDALCO.NS, HINDUNILVR.NS, ICICIBANK.NS, INDUSINDBK.NS, INFY.NS, ITC.NS, JSWSTEEL.NS, KOTAKBANK.NS, LT.NS, LTIM.NS, M&M.NS, MARUTI.NS, NESTLEIND.NS, NTPC.NS, ONGC.NS, POWERGRID.NS, RELIANCE.NS, SBILIFE.NS, SBIN.NS, SUNPHARMA.NS, TATACONSUM.NS, TATAMOTORS.NS, TATASTEEL.NS, TCS.NS, TECHM.NS, TITAN.NS, ULTRACEMCO.NS, WIPRO.NS"

    with st.sidebar:
        st.header("📋 Strategy Parameters")
        u_input = st.text_area("Ticker Universe (Nifty 500)", nifty_500_sample, height=200)
        p_size = st.slider("Portfolio Size", 5, 30, 20)
        p_buffer = st.slider("Exit Rank Buffer", 0, 30, 20)
        p_ret = st.slider("Retracement Max %", 10, 80, 50) / 100
        mkt_ema = st.selectbox("Regime Switch (Nifty EMA)", [20, 200], index=0)
        
        run_btn = st.button("⚡ Run Pro-Tracer Engine")

    if run_btn:
        tickers = [t.strip() for t in u_input.split(",") if t.strip()]
        engine = ProTracerNifty500(tickers)
        
        if engine.fetch_data():
            results = engine.run_engine({
                'size': p_size, 'buffer': p_buffer, 'retracement': p_ret,
                'mkt_filter_ema': mkt_ema, 'lookback': 252
            })
            
            if not results.empty:
                st.success("Analysis Complete")
                
                # --- Metrics Dashboard ---
                col1, col2, col3 = st.columns(3)
                last_row = results.iloc[-1]
                col1.metric("Regime Status", last_row['Mkt_Regime'])
                col2.metric("Active Momentum Picks", last_row['Count'])
                col3.metric("Rebalance Cycles", len(results))
                
                st.subheader("🗓️ Momentum Rebalance History")
                st.dataframe(results, use_container_width=True)
                
                # Current Picks
                st.subheader("🎯 Current Momentum Basket")
                st.write(last_row['Basket'])
            else:
                st.warning("No stocks qualified. Check your filter settings.")

if __name__ == "__main__":
    main()