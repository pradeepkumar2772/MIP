import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

class ProTracerInvestingEngine:
    def __init__(self, universe_tickers, benchmark_ticker='^NSEI'):
        self.tickers = universe_tickers
        self.benchmark = benchmark_ticker
        self.data = None
        self.benchmark_data = None

    def fetch_data(self, start_date, end_date):
        """Fetches adjusted close, high, and low for all tickers."""
        raw_data = yf.download(self.tickers + [self.benchmark], start=start_date, end=end_date)
        self.data = raw_data['Adj Close']
        self.highs = raw_data['High']
        self.lows = raw_data['Low']
        self.benchmark_data = self.data[self.benchmark]
        self.data = self.data.drop(columns=[self.benchmark])

    def calculate_metrics(self, lookback=252):
        """Calculates Volar, RSI, and Trend Filters."""
        # 1. Returns (Performance)
        returns = self.data.pct_change(lookback)
        
        # 2. Volar Calculation (Return / Average Daily Range)
        daily_range = (self.highs - self.lows) / self.data
        avg_volatility = daily_range.rolling(window=lookback).mean()
        volar_scores = returns / avg_volatility
        
        # 3. RSI Calculation (252-period)
        delta = self.data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=lookback).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=lookback).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 4. Moving Averages (Absolute Trend)
        ema_200 = self.data.ewm(span=200, adjust=False).mean()
        
        # 5. Ratio Chart EMA (Relative Trend)
        ratio_chart = self.data.div(self.benchmark_data, axis=0)
        ratio_ema_200 = ratio_chart.ewm(span=200, adjust=False).mean()
        
        # 6. Retracement from 52W High
        high_52w = self.highs.rolling(window=252).max()
        retracement = (high_52w - self.data) / high_52w
        
        return {
            'volar': volar_scores,
            'rsi': rsi,
            'ema_200': ema_200,
            'ratio_ema': ratio_ema_200,
            'ratio_chart': ratio_chart,
            'retracement': retracement,
            'returns': returns
        }

    def run_backtest(self, params):
        """
        Executes the MIP-12/MIP-37 logic with optimizer parameters.
        params = {portfolio_size, buffer, mkt_filter_period, rank_metric, retracement_limit}
        """
        metrics = self.calculate_metrics(params['lookback'])
        
        # Market Filter (Index Trend)
        mkt_ema = self.benchmark_data.ewm(span=params['mkt_filter_period'], adjust=False).mean()
        mkt_bullish = self.benchmark_data > mkt_ema
        
        portfolio = []
        history = []
        dates = self.data.index[params['lookback']:]
        
        # Monthly Rebalance Logic (First trading day of month)
        rebalance_dates = dates[dates.to_series().dt.month != dates.to_series().shift(1).dt.month]
        
        for date in rebalance_dates:
            current_mkt_bullish = mkt_bullish.loc[date]
            
            # 1. Filter Universe
            eligible = (
                (self.data.loc[date] > metrics['ema_200'].loc[date]) &        # Absolute Trend
                (metrics['ratio_chart'].loc[date] > metrics['ratio_ema'].loc[date]) & # Relative Trend
                (metrics['retracement'].loc[date] <= params['retracement_limit']) # Retracement Filter
            )
            
            # 2. Ranking
            if eligible.any():
                candidates = metrics[params['rank_metric']].loc[date][eligible].sort_values(ascending=False)
                
                # 3. Rebalance with Buffer (Exit Rank Logic)
                # Keep existing if rank <= (Size + Buffer)
                new_portfolio = [s for s in portfolio if s in candidates.index and 
                                 list(candidates.index).index(s) < (params['size'] + params['buffer'])]
                
                # 4. Fill to Portfolio Size (Only if Market is Bullish)
                if current_mkt_bullish:
                    needed = params['size'] - len(new_portfolio)
                    top_new = [s for s in candidates.index if s not in new_portfolio][:needed]
                    new_portfolio.extend(top_new)
                
                portfolio = new_portfolio
                history.append({'date': date, 'stocks': list(portfolio), 'mkt_status': current_mkt_bullish})

        return pd.DataFrame(history)

# --- Optimizer Implementation ---
def optimize_pro_tracer(engine):
    param_grid = {
        'size': [10, 20],
        'buffer': [10, 20], # 100% buffer
        'mkt_filter_period': [20, 200],
        'rank_metric': ['volar', 'rsi'],
        'retracement_limit': [0.20, 0.50],
        'lookback': [252]
    }
    
    # In a real run, use itertools.product to loop through these
    # For now, let's run the author's recommended MIP-12
    mip_12_results = engine.run_backtest({
        'size': 20, 'buffer': 20, 'mkt_filter_period': 20,
        'rank_metric': 'volar', 'retracement_limit': 0.50, 'lookback': 252
    })
    return mip_12_results