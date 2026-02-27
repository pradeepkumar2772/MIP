import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Pro-Tracer Momentum Engine", layout="wide")

class ProTracerEngine:
    def __init__(self, tickers, benchmark='^NSEI'):
        self.tickers = [t.strip() for t in tickers]
        self.benchmark = benchmark
        self.raw_data = None
        self.adj_close = None
        self.highs = None
        self.lows = None

    def fetch_data(self, years=5):
        """Fetches data with robust MultiIndex handling for Streamlit Cloud."""
        all_symbols = list(set(self.tickers + [self.benchmark]))
        end_date = datetime.today()
        start_date = end_date - timedelta(days=years*365 + 300) # Buffer for 252 lookback
        
        try:
            # threads=False is more stable in shared cloud environments
            data = yf.download(all_symbols, start=start_date, end=end_date, progress=False, threads=False)
            
            if data.empty:
                return False
            
            # Extracting levels safely
            self.adj_close = data['Adj Close']
            self.highs = data['High']
            self.lows = data['Low']
            
            return True
        except Exception as e:
            st.error(f"Data Download Error: {e}")
            return False

    def run_strategy(self, params):
        """Implements MIP-12 Logic: Volar, Retracement, Trend Filters, and Buffer."""
        # 1. Base Calculations
        lookback = params['lookback']
        
        # Performance (252-period Return)
        returns = self.adj_close.pct_change(lookback)
        
        # Volatility (Avg Daily Fluctuation as per Ch. 5)
        daily_range = (self.highs - self.lows) / self.adj_close
        avg_vol = daily_range.rolling(window=lookback).mean()
        
        # Volar Score (The primary ranking metric)
        volar = returns / avg_vol
        
        # 2. Filters
        # Stock Absolute Trend (Stock > 200-EMA)
        ema_200 = self.adj_close.ewm(span=200, adjust=False).mean()
        
        # Stock Retracement (within 50% of 52W High)
        high_52w = self.highs.rolling(window=252).max()
        within_retracement = self.adj_close >= (high_52w * (1 - params['retracement']))
        
        # Market Trend Filter (Nifty > 20-EMA)
        mkt_series = self.adj_close[self.benchmark]
        mkt_ema = mkt_series.ewm(span=params['mkt_filter'], adjust=False).mean()
        mkt_bullish = mkt_series > mkt_ema
        
        # 3. Simulation Loop
        portfolio = []
        history = []
        
        # Rebalance Monthly (Approx every 21 trading days)
        dates = volar.index[lookback+20:]
        rebalance_dates = dates[::21] 
        
        for date in rebalance_dates:
            curr_mkt = mkt_bullish.loc[date]
            
            # Universe of stocks that pass absolute trend and retracement
            eligible_mask = (self.adj_close.loc[date] > ema_200.loc[date]) & within_retracement.loc[date]
            eligible_stocks = eligible_mask[eligible_mask].index.tolist()
            if self.benchmark in eligible_stocks: eligible_stocks.remove(self.benchmark)
            
            # Calculate Scores for eligible ones
            scores = volar.loc[date, eligible_stocks].sort_values(ascending=False)
            
            # --- Rebalance Logic with Buffer (Ch. 3 & 11) ---
            # 1. Keep existing if rank < (Size + Buffer)
            current_holdings = [s for s in portfolio if s in scores.index and 
                                scores.index.get_loc(s) < (params['size'] + params['buffer'])]
            
            # 2. If Market is Bullish, Fill to 'Size'
            if curr_mkt:
                needed = params['size'] - len(current_holdings)
                new_adds = [s for s in scores.index if s not in current_holdings][:needed]
                current_holdings.extend(new_adds)
            
            portfolio = current_holdings
            history.append({
                'Date': date.date(),
                'Market Bullish': curr_mkt,
                'Holdings Count': len(portfolio),
                'Stocks': ", ".join(portfolio)
            })
            
        return pd.DataFrame(history)

# --- MAIN UI ---
def main():
    st.title("🚀 Pro-Tracer: Momentum Investing Engine")
    st.info("Based on 'Master Momentum Investing' by Prashant Shah")

    with st.sidebar:
        st.header("⚙️ Strategy Blueprint")
        # Pre-set Universe (User can add more)
        raw_tickers = st.text_area("Ticker List (NSE)", "RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS, BHARTIARTL.NS, SBIN.NS,LTIM.NS, TITAN.NS, ADANIENT.NS")
        size = st.slider("Portfolio Size (N)", 5, 20, 10)
        mkt_f = st.selectbox("Market Trend Filter (EMA)", [20, 200], index=0)
        ret_l = st.slider("Retracement Limit (%)", 10, 50, 50)
        
        run = st.button("Generate Rebalance Schedule")

    if run:
        tickers = raw_tickers.split(",")
        engine = ProTracerEngine(tickers)
        
        with st.spinner("Downloading and Decoding Data..."):
            if engine.fetch_data():
                results = engine.run_strategy({
                    'size': size,
                    'buffer': size, # 100% buffer
                    'mkt_filter': mkt_f,
                    'retracement': ret_l/100,
                    'lookback': 252
                })
                
                if not results.empty:
                    st.success("Analysis Complete")
                    
                    # Dashboard Metrics
                    col1, col2 = st.columns(2)
                    col1.metric("Current Regime", "Bullish" if results.iloc[-1]['Market Bullish'] else "Bearish/Pause")
                    col2.metric("Portfolio Churn (Avg)", round(results['Holdings Count'].mean(), 1))
                    
                    st.subheader("🗓️ Monthly Rebalance History")
                    st.dataframe(results, use_container_width=True)
                else:
                    st.warning("No stocks met the criteria. Try loosening filters.")
            else:
                st.error("Could not fetch data. Ensure Tickers are correct and end in .NS")

if __name__ == "__main__":
    main()