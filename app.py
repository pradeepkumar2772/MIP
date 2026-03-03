import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime

# --- UI Setup ---
st.title("📜 Pro-Tracer: Detailed Trade Log")

# Inputs for the SPECIFIC parameters you want to inspect
col1, col2, col3 = st.columns(3)
ticker = col1.text_input("Ticker", "RELIANCE.NS")
best_rsi = col2.number_input("RSI Length", value=14)
best_ema = col3.number_input("EMA Length (Average Line)", value=9)

start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))

if st.button("Generate Trade Book"):
    df = yf.download(ticker, start=start_date)
    df.columns = df.columns.get_level_values(0)
    
    # 1. Indicator Calculation
    df['RSI'] = ta.rsi(df['Close'], length=best_rsi)
    df['RSI_EMA'] = ta.ema(df['RSI'], length=best_ema)
    
    # 2. Identify Signals
    # 1 = Long, 0 = Cash
    df['Signal'] = (df['RSI'] > df['RSI_EMA']).astype(int)
    df['Position_Change'] = df['Signal'].diff()

    # 3. Extract Individual Trades
    trades = []
    in_trade = False
    entry_date = None
    entry_price = 0

    for i in range(1, len(df)):
        # Entry Logic (Crossover Up)
        if df['Position_Change'].iloc[i] == 1 and not in_trade:
            in_trade = True
            entry_date = df.index[i]
            entry_price = df['Close'].iloc[i]
            
        # Exit Logic (Crossover Down)
        elif df['Position_Change'].iloc[i] == -1 and in_trade:
            in_trade = False
            exit_date = df.index[i]
            exit_price = df['Close'].iloc[i]
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            
            trades.append({
                "Entry Date": entry_date.date(),
                "Exit Date": exit_date.date(),
                "Entry Price": round(entry_price, 2),
                "Exit Price": round(exit_price, 2),
                "P&L %": round(pnl_pct, 2)
            })

    # 4. Display the Trade Book
    if trades:
        trade_df = pd.DataFrame(trades)
        st.write(f"### Trade Ledger for RSI({best_rsi}) vs EMA({best_ema})")
        st.dataframe(trade_df)
        
        # Summary Stats
        win_rate = (trade_df['P&L %'] > 0).mean() * 100
        st.metric("Total Trades", len(trade_df))
        st.metric("Win Rate", f"{win_rate:.2f}%")
    else:
        st.warning("No trades were triggered with these parameters in this date range.")