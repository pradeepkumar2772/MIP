import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v2.3", layout="wide")

st.title("🛡️ Pro-Tracer: Visual Risk Engine")
st.sidebar.header("Global Settings")

# --- Global Inputs ---
ticker = st.sidebar.text_input("Ticker Symbol", "RELIANCE.NS")
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
        # ... [Optimizer Code remains same as stable version] ...
        st.header("🔍 Brute-Force Parameter Optimizer")
        if st.button("🚀 Run Full Optimization"):
            results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            rsi_range = range(3, 253, 10) # Step 10 for faster web demo
            ema_range = range(3, 51, 5)
            progress_bar = st.progress(0)
            total_steps = len(rsi_range)
            for i, r_len in enumerate(rsi_range):
                progress_bar.progress(i / total_steps)
                rsi_series = ta.rsi(df['Close'], length=r_len)
                for e_len in ema_range:
                    rsi_ema = ta.ema(rsi_series, length=e_len)
                    signal = (rsi_series > rsi_ema).astype(int)
                    strat_ret = (df['Market_Ret'] * signal.shift(1)).fillna(0)
                    cum_ret = (1 + strat_ret).cumprod()
                    total_return = cum_ret.iloc[-1] - 1
                    max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
                    results.append({'RSI_Len': r_len, 'EMA_Len': e_len, 'ROI %': round(total_return * 100, 2), 'Max_DD %': round(max_dd * 100, 2)})
            st.success("Optimization Complete!")
            st.dataframe(pd.DataFrame(results).sort_values('ROI %', ascending=False).head(20))

    elif mode == "Trade Detailer":
        st.header("📜 Trade Detailer & Exit Analysis")
        col1, col2, col3 = st.columns(3)
        in_rsi = col1.number_input("RSI Look-back", value=14, min_value=3)
        in_ema = col2.number_input("EMA (Average Line)", value=9, min_value=3)
        stop_loss_pct = col3.number_input("Stop Loss %", value=5.0, min_value=0.0, step=0.5)
        
        if st.button("📊 Generate Quant Report"):
            df = df_raw.copy()
            df['RSI'] = ta.rsi(df['Close'], length=in_rsi)
            df['RSI_EMA'] = ta.ema(df['RSI'], length=in_ema)
            
            trades = []
            in_trade = False
            entry_price = 0

            for i in range(1, len(df)):
                current_price = float(df['Close'].iloc[i])
                rsi_val = df['RSI'].iloc[i]
                ema_val = df['RSI_EMA'].iloc[i]
                prev_rsi = df['RSI'].iloc[i-1]
                prev_ema = df['RSI_EMA'].iloc[i-1]

                if not in_trade:
                    if rsi_val > ema_val and prev_rsi <= prev_ema:
                        in_trade, entry_date, entry_price = True, df.index[i], current_price
                elif in_trade:
                    sl_triggered = stop_loss_pct > 0 and current_price <= entry_price * (1 - stop_loss_pct/100)
                    rsi_cross = rsi_val < ema_val
                    
                    if sl_triggered or rsi_cross:
                        in_trade = False
                        exit_price = entry_price * (1 - stop_loss_pct/100) if sl_triggered else current_price
                        reason = "Stop Loss" if sl_triggered else "RSI Crossover"
                        pnl = ((exit_price - entry_price) / entry_price) * 100
                        trades.append({"Entry Date": entry_date.date(), "Exit Date": df.index[i].date(), 
                                       "Entry Price": round(entry_price, 2), "Exit Price": round(exit_price, 2), 
                                       "P&L %": round(pnl, 2), "Exit Reason": reason})

            if trades:
                t_df = pd.DataFrame(trades)
                
                # Layout for Charts
                chart_col1, chart_col2 = st.columns(2)

                # --- Exit Reason Pie Chart ---
                with chart_col1:
                    st.write("### Exit Reason Breakdown")
                    exit_counts = t_df['Exit Reason'].value_counts()
                    fig, ax = plt.subplots()
                    ax.pie(exit_counts, labels=exit_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
                    ax.axis('equal') 
                    st.pyplot(fig)
                
                # --- P&L Distribution ---
                with chart_col2:
                    st.write("### P&L % per Trade")
                    st.bar_chart(t_df['P&L %'])

                # Full Ledger
                st.write("### Detailed Trade Ledger")
                st.dataframe(t_df)
            else:
                st.warning("No trades found.")