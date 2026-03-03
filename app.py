import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v2.1", layout="wide")

st.title("🛡️ Pro-Tracer: Risk-Managed Engine")
st.sidebar.header("Global Settings")

# --- Global Inputs ---
ticker = st.sidebar.text_input("Ticker Symbol", "TRENT.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(1990, 1, 1), min_value=datetime.date(1990, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

mode = st.sidebar.radio("Select Module", ["Brute-Force Optimizer", "Trade Detailer"])

@st.cache_data
def get_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return data
    except Exception as e:
        st.error(f"Data error: {e}")
    return pd.DataFrame()

df_raw = get_data(ticker, start_date, end_date)

if df_raw.empty:
    st.warning("Awaiting data...")
else:
    if mode == "Brute-Force Optimizer":
        st.header("🔍 Optimizer")
        st.info("Stop Loss optimization coming soon. Currently optimizing RSI/EMA core.")
        # ... [Previous Optimizer Code remains the same for the core logic] ...

    elif mode == "Trade Detailer":
        st.header("📜 Trade Detailer with Stop Loss")
        
        c1, c2, c3 = st.columns(3)
        in_rsi = c1.number_input("RSI Length", value=14)
        in_ema = c2.number_input("EMA (Average Line)", value=9)
        sl_pct = c3.number_input("Stop Loss % (0 to disable)", value=5.0)
        
        if st.button("📊 Run Backtest"):
            df = df_raw.copy()
            df['RSI'] = ta.rsi(df['Close'], length=in_rsi)
            df['RSI_EMA'] = ta.ema(df['RSI'], length=in_ema)
            
            trades = []
            in_trade = False
            entry_price = 0
            
            # Logic Loop with Stop Loss
            for i in range(1, len(df)):
                current_price = float(df['Close'].iloc[i])
                rsi_val = df['RSI'].iloc[i]
                ema_val = df['RSI_EMA'].iloc[i]
                
                # ENTRY: RSI crosses above EMA
                if not in_trade:
                    if rsi_val > ema_val and df['RSI'].iloc[i-1] <= df['RSI_EMA'].iloc[i-1]:
                        in_trade = True
                        entry_date = df.index[i]
                        entry_price = current_price
                        stop_price = entry_price * (1 - (sl_pct / 100))
                
                # EXIT: Either RSI crosses below EMA OR Stop Loss is hit
                elif in_trade:
                    hit_sl = sl_pct > 0 and current_price <= stop_price
                    rsi_exit = rsi_val < ema_val
                    
                    if hit_sl or rsi_exit:
                        in_trade = False
                        exit_date = df.index[i]
                        exit_price = stop_price if hit_sl else current_price
                        exit_type = "STOP LOSS" if hit_sl else "RSI CROSS"
                        
                        pnl = ((exit_price - entry_price) / entry_price) * 100
                        trades.append({
                            "Entry Date": entry_date.date(),
                            "Exit Date": exit_date.date(),
                            "Entry Price": round(entry_price, 2),
                            "Exit Price": round(exit_price, 2),
                            "P&L %": round(pnl, 2),
                            "Exit Reason": exit_type
                        })

            if trades:
                t_df = pd.DataFrame(trades)
                
                # Metrics
                win_rate = (t_df['P&L %'] > 0).mean()
                total_ret = (t_df['P&L %']/100 + 1).prod() - 1
                
                st.subheader("📊 Quant Scorecard")
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Return", f"{total_ret*100:.2f}%")
                m2.metric("Win Rate", f"{win_rate*100:.1f}%")
                m3.metric("Stop Loss Trades", len(t_df[t_df['Exit Reason'] == "STOP LOSS"]))
                
                st.write("### Detailed Trade Ledger")
                st.dataframe(t_df)
                
                st.write("### Equity Curve (Compounded)")
                strat_cum = (t_df['P&L %']/100 + 1).cumprod()
                st.line_chart(strat_cum)