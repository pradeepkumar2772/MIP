import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st

# 1. Setup & Data
ticker = "RELIANCE.NS"
data = yf.download(ticker, start="2022-01-01")

# 2. Calculate Turtle Parameters
# System 1
data['S1_High'] = data['High'].rolling(window=20).max().shift(1)
data['S1_Low'] = data['Low'].rolling(window=10).min().shift(1)
# System 2
data['S2_High'] = data['High'].rolling(window=55).max().shift(1)
data['S2_Low'] = data['Low'].rolling(window=20).min().shift(1)
# Risk Management (ATR)
data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=20)

# 3. Simulation Variables
balance = 1000000  # ₹10 Lakhs
risk_pct = 0.02    # 2% Risk
position = 0
entry_price = 0
last_trade_won = False
equity_curve = []

# 4. The Turtle Loop
for i in range(len(data)):
    row = data.iloc[i]
    current_close = row['Close']
    
    if position == 0:  # Looking for Entry
        # Entry Logic for System 1 (with winner filter)
        if not last_trade_won and row['High'] > row['S1_High']:
            # Calculate Unit Size
            stop_dist = 2 * row['ATR']
            unit_size = (balance * risk_pct) // stop_dist
            position = unit_size
            entry_price = row['S1_High']
            
        # Entry Logic for System 2 (Backup if S1 was skipped)
        elif row['High'] > row['S2_High']:
            stop_dist = 2 * row['ATR']
            unit_size = (balance * risk_pct) // stop_dist
            position = unit_size
            entry_price = row['S2_High']
            
    elif position > 0:  # Looking for Exit (Long)
        # Exit if price hits 10-day low (S1) or 2-ATR Stop Loss
        stop_loss = entry_price - (2 * row['ATR'])
        if row['Low'] < row['S1_Low'] or row['Low'] < stop_loss:
            exit_price = min(row['S1_Low'], stop_loss)
            pnl = (exit_price - entry_price) * position
            balance += pnl
            last_trade_won = pnl > 0
            position = 0
            
    equity_curve.append(balance)

data['Equity'] = equity_curve