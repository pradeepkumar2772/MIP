import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --- SETTINGS ---
st.set_page_config(page_title="Pro-Tracer Full Engine", layout="wide")

class ProTracerFullEngine:
    def __init__(self, tickers, benchmark='^NSEI'):
        # Ensure all tickers are stripped of whitespace and are in a list
        self.tickers = [t.strip() for t in tickers if t.strip()]
        self.benchmark = benchmark
        self.adj_close = None
        self.highs = None
        self.lows = None

    def fetch_data(self, years=5):
        # Explicitly include benchmark in the download list
        all_symbols = list(set(self.tickers + [self.benchmark]))
        start_date = datetime.today() - timedelta(days=years*365 + 400) # Added extra buffer for indicators
        
        try:
            raw = yf.download(all_symbols, start=start_date, progress=False, threads=False)
            
            if raw.empty:
                st.error("Yahoo Finance returned an empty dataset.")
                return False
            
            # --- ROBUST MULTI-INDEX FLATTENING ---
            # This converts ('Adj Close', 'TCS.NS') into a single dataframe where columns are just 'TCS.NS'
            self.adj_close = raw['Adj Close']
            self.highs = raw['High']
            self.lows = raw['Low']
            
            # Check if benchmark actually exists in the columns
            if self.benchmark not in self.adj_close.columns:
                st.error(f"Benchmark {self.benchmark} was not found in the downloaded data. Columns available: {list(self.adj_close.columns)}")
                return False
                
            return True
        except Exception as e:
            st.error(f"Download or Data Structuring Error: {e}")
            return False

    def run_engine(self, p):
        # Local references for cleaner math
        df = self.adj_close
        hi = self.highs
        lo = self.lows
        
        # SAFE ACCESS to Benchmark
        bm = df[self.benchmark]
        
        # 1. BASE CALCULATIONS (Chapter 3-6)
        lookback = p['lookback']
        ret_252 = df.pct_change(lookback)
        
        # Volar Calculation: Return / Avg Daily Range
        daily_range = (hi - lo) / df
        volar = ret_252 / daily_range.rolling(lookback).mean()
        
        # Absolute Trend (EMA 200)
        ema_200 = df.ewm(span=200, adjust=False).mean()
        
        # Relative Trend (Ratio EMA 200)
        # We divide every stock by the benchmark series
        ratio = df.div(bm, axis=0)
        ratio_ema = ratio.ewm(span=200, adjust=False).mean()
        
        # Retracement (52W High)
        h52w = hi.rolling(lookback).max()
        pct_from_high = (h52w - df) / h52w
        
        # RSI 252 (Tadka - Chapter 6)
        delta = df.diff()
        gain = (delta.where(delta > 0, 0)).rolling(lookback).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(lookback).mean()
        rsi = 100 - (100 / (1 + (gain / (loss + 1e-9)))) # Added epsilon to prevent div by zero
        
        # MACD (100, 200)
        macd_line = df.ewm(span=100).mean() - df.ewm(span=200).mean()

        # 2. MARKET REGIME FILTER (Chapter 5)
        mkt_ema = bm.ewm(span=p['mkt_filter_ema']).mean()
        mkt_bullish = bm > mkt_ema

        # 3. PORTFOLIO SIMULATION
        history = []
        portfolio = []
        # Find rebalance dates (roughly monthly)
        valid_dates = df.index[lookback + 20:]
        rebalance_dates = valid_dates[::21] 

        for date in rebalance_dates:
            is_bullish = mkt_bullish.loc[date]
            
            # Logic Filters
            # Note: We drop the benchmark from eligible candidates
            current_universe = [t for t in df.columns if t != self.benchmark]
            
            # Vectorized eligibility for the specific date
            eligibility = (
                (df.loc[date] > ema_200.loc[date]) & 
                (ratio.loc[date] > ratio_ema.loc[date]) & 
                (pct_from_high.loc[date] <= p['retracement'])
            )
            
            if p['use_rsi_filter']:
                eligibility &= (rsi.loc[date] > 50)
            if p['use_macd_filter']:
                eligibility &= (macd_line.loc[date] > 0)
            
            eligible_stocks = eligibility[eligibility].index.tolist()
            if self.benchmark in eligible_stocks: eligible_stocks.remove(self.benchmark)
            
            # Ranking
            if not eligible_stocks:
                scores = pd.Series(dtype=float)
            else:
                scores = volar.loc[date, eligible_stocks].sort_values(ascending=False)

            # --- REBALANCE LOGIC (Chapter 11) ---
            # 1. Keep existing if rank <= (Size + Buffer)
            # Use get_loc safely
            temp_portfolio = []
            for s in portfolio:
                if s in scores.index:
                    rank = scores.index.get_loc(s)
                    if rank < (p['size'] + p['buffer']):
                        temp_portfolio.append(s)
            
            portfolio = temp_portfolio

            # 2. Add new leaders only if Market is Bullish
            if is_bullish:
                needed = p['size'] - len(portfolio)
                new_candidates = [s for s in scores.index if s not in portfolio]
                portfolio.extend(new_candidates[:needed])
            
            history.append({
                'Date': date.date(),
                'Nifty Trend': "BULLISH" if is_bullish else "BEARISH (Wait)",
                'Active Stocks': len(portfolio),
                'Portfolio': ", ".join(portfolio)
            })

        return pd.DataFrame(history)

# --- APP UI ---
def main():
    st.title("🚀 Pro-Tracer: Complete Strategy Blueprint")
    st.markdown("Implements MIP-12 + Tadka Filters + Execution Buffer")

    with st.sidebar:
        st.header("📋 Configuration")
        u_input = st.text_area("Ticker List", "RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS, BHARTIARTL.NS, SBIN.NS, LTIM.NS, TITAN.NS, SUNPHARMA.NS, TATAMOTORS.NS, ONGC.NS, NTPC.NS, COALINDIA.NS")
        
        with st.expander("Core Strategy Settings"):
            p_size = st.slider("Target Portfolio Size", 5, 20, 10)
            p_buffer = st.slider("Buffer (Extra Ranks)", 0, 20, 5)
            p_ret = st.slider("Retracement Max %", 10, 80, 50) / 100
            mkt_ema = st.selectbox("Regime Filter (Index EMA)", [20, 200], index=0)
        
        with st.expander("Tadka Filters"):
            use_rsi = st.checkbox("252-RSI > 50", value=True)
            use_macd = st.checkbox("MACD (100,200) Bullish", value=False)
            
        run_btn = st.button("🔥 Run Momentum Engine")

    if run_btn:
        tickers = u_input.split(",")
        engine = ProTracerFullEngine(tickers)
        
        with st.spinner("Decoding markets..."):
            if engine.fetch_data():
                res = engine.run_engine({
                    'size': p_size, 'buffer': p_buffer, 'retracement': p_ret,
                    'mkt_filter_ema': mkt_ema, 'use_rsi_filter': use_rsi,
                    'use_macd_filter': use_macd, 'lookback': 252
                })
                
                if not res.empty:
                    st.success("Analysis Complete!")
                    st.subheader("🗓️ Monthly Rebalance Schedule")
                    st.dataframe(res, use_container_width=True)
                else:
                    st.warning("No data generated. Loosen your filters.")

if __name__ == "__main__":
    main()