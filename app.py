import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --- SETTINGS ---
st.set_page_config(page_title="Pro-Tracer Full Engine", layout="wide")

class ProTracerFullEngine:
    def __init__(self, tickers, benchmark='^NSEI'):
        self.tickers = [t.strip() for t in tickers]
        self.benchmark = benchmark
        self.data = None

    def fetch_data(self, years=5):
        all_symbols = list(set(self.tickers + [self.benchmark]))
        start_date = datetime.today() - timedelta(days=years*365 + 300)
        try:
            raw = yf.download(all_symbols, start=start_date, progress=False, threads=False)
            if raw.empty: return False
            self.adj_close = raw['Adj Close']
            self.highs = raw['High']
            self.lows = raw['Low']
            return True
        except Exception as e:
            st.error(f"Download Error: {e}")
            return False

    def run_engine(self, p):
        """
        p: parameters dictionary containing all UI selections
        """
        df = self.adj_close
        hi = self.highs
        lo = self.lows
        bm = df[self.benchmark]
        
        # 1. BASE CALCULATIONS (Chapter 3-6)
        # Returns & Volar
        ret_252 = df.pct_change(p['lookback'])
        daily_range = (hi - lo) / df
        volar = ret_252 / daily_range.rolling(p['lookback']).mean()
        
        # Absolute Trend (EMA 200)
        ema_200 = df.ewm(span=200, adjust=False).mean()
        
        # Relative Trend (Ratio EMA 200)
        ratio = df.div(bm, axis=0)
        ratio_ema = ratio.ewm(span=200, adjust=False).mean()
        
        # Retracement (52W High)
        h52w = hi.rolling(252).max()
        pct_from_high = (h52w - df) / h52w
        
        # Momentum Indicators (Tadka - Chapter 6)
        # RSI 252
        delta = df.diff()
        gain = (delta.where(delta > 0, 0)).rolling(p['lookback']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(p['lookback']).mean()
        rsi = 100 - (100 / (1 + (gain / loss)))
        
        # MACD (100, 200)
        macd_line = df.ewm(span=100).mean() - df.ewm(span=200).mean()

        # 2. MARKET FILTER (Chapter 5 & 7)
        mkt_ema = bm.ewm(span=p['mkt_filter_ema']).mean()
        mkt_bullish = bm > mkt_ema

        # 3. PORTFOLIO SIMULATION
        history = []
        portfolio = []
        rebalance_dates = df.index[p['lookback'] + 20 :: 21] # Monthly

        for date in rebalance_dates:
            is_bullish = mkt_bullish.loc[date]
            
            # Applying All Logic Filters (The "Blueprint")
            eligible_mask = (
                (df.loc[date] > ema_200.loc[date]) &             # Absolute Trend
                (ratio.loc[date] > ratio_ema.loc[date]) &        # Relative Trend
                (pct_from_high.loc[date] <= p['retracement'])    # Retracement
            )
            
            # Optional Tadka Filters
            if p['use_rsi_filter']:
                eligible_mask &= (rsi.loc[date] > 50)
            if p['use_macd_filter']:
                eligible_mask &= (macd_line.loc[date] > 0)
                
            eligible_stocks = eligible_mask[eligible_mask].index.tolist()
            if self.benchmark in eligible_stocks: eligible_stocks.remove(self.benchmark)
            
            # Ranking (Chapter 9)
            if not eligible_stocks:
                scores = pd.Series()
            else:
                scores = volar.loc[date, eligible_stocks].sort_values(ascending=False)

            # Rebalance Logic with Buffer (Chapter 11)
            # 1. Sell if Rank > (Size + Buffer)
            portfolio = [s for s in portfolio if s in scores.index and 
                         scores.index.get_loc(s) < (p['size'] + p['buffer'])]
            
            # 2. Buy only if Market is Bullish
            if is_bullish:
                needed = p['size'] - len(portfolio)
                new_entries = [s for s in scores.index if s not in portfolio][:needed]
                portfolio.extend(new_entries)
            
            history.append({
                'Date': date.date(),
                'Regime': "Bullish" if is_bullish else "Bearish (Cash)",
                'Count': len(portfolio),
                'Stocks': ", ".join(portfolio)
            })

        return pd.DataFrame(history)

# --- APP UI ---
def main():
    st.title("🚀 Pro-Tracer: Full Momentum Strategy Engine")
    st.sidebar.header("📋 Strategy Configuration")

    # Sidebar Parameters
    u_tickers = st.sidebar.text_area("Ticker Universe", "RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS, BHARTIARTL.NS, SBIN.NS, LTIM.NS, TITAN.NS, ADANIENT.NS, MARUTI.NS, AXISBANK.NS, SUNPHARMA.NS, TATAMOTORS.NS, ONGC.NS")
    
    with st.sidebar.expander("1. Core Engine (Ch. 3-5)"):
        p_size = st.slider("Portfolio Size", 5, 20, 10)
        p_buffer = st.slider("Exit Rank Buffer", 0, 20, 10)
        p_ret = st.slider("Retracement (%)", 10, 50, 50) / 100
        mkt_ema = st.selectbox("Market Filter EMA", [20, 200], index=0)

    with st.sidebar.expander("2. Tadka Strategy (Ch. 6)"):
        use_rsi = st.checkbox("RSI 252 > 50 Filter", value=True)
        use_macd = st.checkbox("MACD (100,200) > 0 Filter", value=False)

    if st.sidebar.button("⚙️ Run Momentum Engine"):
        engine = ProTracerFullEngine(u_tickers.split(","))
        with st.spinner("Processing Blueprints..."):
            if engine.fetch_data():
                res = engine.run_engine({
                    'size': p_size, 'buffer': p_buffer, 'retracement': p_ret,
                    'mkt_filter_ema': mkt_ema, 'use_rsi_filter': use_rsi,
                    'use_macd_filter': use_macd, 'lookback': 252
                })
                
                # --- RESULTS DISPLAY ---
                st.success("Analysis Complete")
                
                # Dashboard Summary
                m1, m2, m3 = st.columns(3)
                m1.metric("Current Market", res.iloc[-1]['Regime'])
                m2.metric("Avg Portfolio Churn", round(res['Count'].mean(), 1))
                m3.metric("Total Rebalances", len(res))

                st.subheader("🗓️ Execution Blueprint (Rebalance History)")
                st.dataframe(res, use_container_width=True)
                
                st.download_button("Download Schedule", res.to_csv().encode('utf-8'), "rebalance_schedule.csv")

if __name__ == "__main__":
    main()