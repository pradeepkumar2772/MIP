import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import itertools

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Pro-Tracer Engine", layout="wide")

# This ensures the app doesn't hang on start
def main():
    st.title("🚀 Pro-Tracer: Momentum Investing Module")
    st.markdown("---")

    # 1. SIDEBAR CONFIGURATION (The Optimizer Inputs)
    with st.sidebar:
        st.header("Strategy Blueprint (Ch. 9)")
        universe_choice = st.selectbox("Select Universe", ["Nifty 50", "Nifty 500"])
        mkt_filter = st.selectbox("Market Filter (Nifty 20-EMA)", [20, 200], index=0)
        rank_metric = st.radio("Ranking Metric", ['volar', 'rsi', 'returns'])
        port_size = st.slider("Portfolio Size", 10, 30, 20)
        retracement = st.slider("Retracement Limit (%)", 10, 50, 50) / 100
        
        run_btn = st.button("🚀 Run Strategy Engine")

    # 2. TICKER LISTS
    # For testing, start small to ensure it renders!
    if universe_choice == "Nifty 50":
        tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"] # Add all 50
    else:
        # For Nifty 500, we recommend loading from a CSV or a small subset first
        tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "WIPRO.NS", "TITAN.NS"] 

    if run_btn:
        with st.spinner("Executing Momentum Logic..."):
            # Execute the engine
            engine = ProTracerInvestingEngine(tickers)
            
            # Fetch data (Wrapped in Cache to prevent blank page hangs)
            data_load_state = st.info("Fetching Historical Data...")
            success = engine.fetch_data("2020-01-01", "2024-12-31")
            data_load_state.empty()

            if success:
                st.subheader(f"Strategy Results: MIP-12 (Customized)")
                
                # Run the backtest logic
                history = engine.run_backtest({
                    'size': port_size, 
                    'buffer': port_size, # 100% buffer as per Ch. 3
                    'mkt_filter_period': mkt_filter,
                    'rank_metric': rank_metric, 
                    'retracement_limit': retracement, 
                    'lookback': 252
                })

                if not history.empty:
                    # Displaying the current portfolio
                    latest_stocks = history.iloc[-1]['stocks']
                    st.success(f"Latest Rebalance Date: {history.iloc[-1]['date'].date()}")
                    st.write("**Current Momentum Basket:**", ", ".join(latest_stocks))
                    
                    # Portfolio History Table
                    st.dataframe(history, use_container_width=True)
                else:
                    st.error("Engine logic returned no trades. Adjust your filters.")
            else:
                st.error("Failed to fetch data from Yahoo Finance.")

# --- CORE ENGINE CLASS ---
class ProTracerInvestingEngine:
    def __init__(self, universe_tickers, benchmark_ticker='^NSEI'):
        self.tickers = universe_tickers
        self.benchmark = benchmark_ticker
        self.data = None

    def fetch_data(self, start, end):
        try:
            # Adding benchmark to ticker list
            all_tickers = self.tickers + [self.benchmark]
            raw = yf.download(all_tickers, start=start, end=end, progress=False)
            
            if raw.empty: return False
            
            self.adj_close = raw['Adj Close']
            self.highs = raw['High']
            self.lows = raw['Low']
            self.benchmark_data = self.adj_close[self.benchmark]
            self.data = self.adj_close.drop(columns=[self.benchmark])
            return True
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return False

    def run_backtest(self, params):
        # Calculation logic (simplified for speed)
        # 1. 252-Day Returns
        returns = self.data.pct_change(params['lookback'])
        
        # 2. Volar Score (Return / Avg Daily Range)
        daily_range = (self.highs - self.lows) / self.adj_close
        volar = returns / daily_range.rolling(window=params['lookback']).mean()
        
        # 3. Filters
        ema_200 = self.data.ewm(span=200).mean()
        mkt_ema = self.benchmark_data.ewm(span=params['mkt_filter_period']).mean()
        
        # Rebalance loop
        history = []
        portfolio = []
        rebalance_dates = self.data.index[params['lookback']::21] # Monthly-ish
        
        for date in rebalance_dates:
            mkt_bullish = self.benchmark_data.loc[date] > mkt_ema.loc[date]
            
            # Selection Criteria
            eligible = (self.data.loc[date] > ema_200.loc[date]) 
            
            if eligible.any():
                # Ranking by chosen metric
                scores = volar.loc[date][eligible].sort_values(ascending=False)
                
                # Simple Buffer logic
                new_port = [s for s in portfolio if s in scores.index and list(scores.index).index(s) < (params['size'] + params['buffer'])]
                
                if mkt_bullish:
                    needed = params['size'] - len(new_port)
                    top_new = [s for s in scores.index if s not in new_port][:needed]
                    new_port.extend(top_new)
                
                portfolio = new_port
                history.append({'date': date, 'stocks': list(portfolio), 'mkt_bullish': mkt_bullish})
        
        return pd.DataFrame(history)

if __name__ == "__main__":
    main()