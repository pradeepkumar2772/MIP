import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st

st.title("Pro-Tracer: Turtle System 1 & 2")

ticker = "RELIANCE.NS"
# Download and flatten columns to avoid the 'Identically-labeled Series' error
df = yf.download(ticker, start="2022-01-01")
df.columns = df.columns.get_level_values(0) # Flattens multi-index if present

if not df.empty:
    # 1. Calculations
    df['S1_High'] = df['High'].rolling(window=20).max().shift(1)
    df['S1_Low'] = df['Low'].rolling(window=10).min().shift(1)
    df['S2_High'] = df['High'].rolling(window=55).max().shift(1)
    df['S2_Low'] = df['Low'].rolling(window=20).min().shift(1)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=20)

    # 2. Simulation State
    balance = 1000000
    risk_pct = 0.02
    position = 0
    units_held = 0
    entry_price = 0
    last_trade_won = False
    equity_curve = []

    # 3. Execution Loop
    for i in range(len(df)):
        # Convert to float to avoid Pandas Series comparison errors
        high = float(df['High'].iloc[i])
        low = float(df['Low'].iloc[i])
        close = float(df['Close'].iloc[i])
        s1_h = float(df['S1_High'].iloc[i])
        s1_l = float(df['S1_Low'].iloc[i])
        s2_h = float(df['S2_High'].iloc[i])
        atr = float(df['ATR'].iloc[i])

        if units_held == 0:
            # Check System 1 (If last trade wasn't a win)
            if not last_trade_won and high > s1_h:
                entry_price = s1_h
                stop_dist = 2 * atr if atr > 0 else 0.01
                unit_qty = (balance * risk_pct) // stop_dist
                units_held = 1
                pos_qty = unit_qty
            # Check System 2 (Always valid)
            elif high > s2_h:
                entry_price = s2_h
                stop_dist = 2 * atr if atr > 0 else 0.01
                unit_qty = (balance * risk_pct) // stop_dist
                units_held = 1
                pos_qty = unit_qty
        
        else:
            # Pyramiding: Add units every 0.5 * ATR move up (Max 4 units)
            if units_held < 4 and high > (entry_price + (0.5 * atr * units_held)):
                units_held += 1
                pos_qty += unit_qty
            
            # Exit Logic: 10-day low OR 2-ATR Stop Loss
            stop_loss = entry_price - (2 * atr)
            if low < s1_l or low < stop_loss:
                exit_price = min(s1_l, stop_loss)
                pnl = (exit_price - entry_price) * pos_qty
                balance += pnl
                last_trade_won = pnl > 0
                units_held = 0
                pos_qty = 0

        equity_curve.append(balance)

    df['Equity'] = equity_curve
    st.line_chart(df['Equity'])
    st.write(f"Final Balance: ₹{balance:,.2f}")