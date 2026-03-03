import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v3.5", layout="wide")

st.title("🛡️ Pro-Tracer: Visual Sensitivity Optimizer")
st.sidebar.header("Global Settings")

ticker = st.sidebar.text_input("Ticker Symbol", "TRENT.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(1999, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

mode = st.sidebar.radio("Select Module", ["Visual Optimizer", "Trade Detailer"])

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
    if mode == "Visual Optimizer":
        st.header("📈 RSI Sensitivity Analysis (3 to 252)")
        st.info("We are scanning every 'Momentum Cycle' to find where this stock thrives.")
        
        if st.button("🚀 Run Deep Sensitivity Scan"):
            results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            
            # Range 3-252, Step 5 for speed
            rsi_range = range(3, 253, 5) 
            progress_bar = st.progress(0)
            
            for i, r_len in enumerate(rsi_range):
                progress_bar.progress((i + 1) / len(rsi_range))
                rsi = ta.rsi(df['Close'], length=r_len)
                
                # Fast Vectorized Simulation
                sig = pd.Series(0, index=df.index)
                in_pos = False
                # Note: For massive ranges, we use a loop to respect the 60/50 state logic
                for j in range(1, len(df)):
                    if not in_pos and rsi.iloc[j] > 60: in_pos = True
                    elif in_pos and rsi.iloc[j] < 50: in_pos = False
                    sig.iloc[j] = 1 if in_pos else 0
                
                strat_ret = (df['Market_Ret'] * sig.shift(1)).fillna(0)
                cum_ret = (1 + strat_ret).cumprod()
                total_return = (cum_ret.iloc[-1] - 1) * 100
                
                results.append({'RSI_Len': r_len, 'ROI': total_return})
            
            res_df = pd.DataFrame(results)
            
            # --- Sensitivity Chart ---
            st.write("### The Momentum Sensitivity Curve")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(res_df['RSI_Len'], res_df['ROI'], marker='o', linestyle='-', color='#1f77b4')
            ax.set_xlabel("RSI Look-back Period (Days)")
            ax.set_ylabel("Total ROI (%)")
            ax.set_title(f"Profit Sensitivity for {ticker}")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.success("Scan Complete!")
            st.write("### Ranked Results")
            st.dataframe(res_df.sort_values('ROI', ascending=False))

    elif mode == "Trade Detailer":
        # ... [Rest of Trade Detailer code from v3.3 remains stable here] ...
        st.header("📜 Trade Detailer & Recovery Analysis")
        col1, col2, col3, col4 = st.columns(4)
        in_rsi = col1.number_input("RSI Look-back", value=14)
        vol_mult = col2.number_input("Vol Spike (x Avg)", value=1.5)
        vol_ma = col3.number_input("Vol Avg Period", value=20)
        stop_loss_pct = col4.number_input("Stop Loss %", value=5.0)
        
        if st.button("📊 Generate Detailed Report"):
            df = df_raw.copy()
            df['RSI'] = ta.rsi(df['Close'], length=in_rsi)
            df['Vol_MA'] = df['Volume'].rolling(window=vol_ma).mean()
            df['Trend_EMA'] = ta.ema(df['Close'], length=200)
            
            trades = []
            in_trade = False
            for i in range(1, len(df)):
                curr_p = float(df['Close'].iloc[i])
                rsi_v = df['RSI'].iloc[i]
                prev_rsi = df['RSI'].iloc[i-1]
                vol_spike = (df['Volume'].iloc[i] > (df['Vol_MA'].iloc[i] * vol_mult)) if not pd.isna(df['Vol_MA'].iloc[i]) else False
                
                if not in_trade:
                    if rsi_v > 60 and prev_rsi <= 60 and curr_p > df['Trend_EMA'].iloc[i] and vol_spike:
                        in_trade, entry_date, entry_price = True, df.index[i], curr_p
                elif in_trade:
                    sl_hit = stop_loss_pct > 0 and curr_p <= entry_price * (1 - stop_loss_pct/100)
                    rsi_exit = rsi_v < 50 and prev_rsi >= 50
                    if sl_hit or rsi_exit:
                        in_trade = False
                        exit_p = entry_price * (1 - stop_loss_pct/100) if sl_hit else curr_p
                        pnl = ((exit_p - entry_price) / entry_price) * 100
                        trades.append({"Entry Date": entry_date.date(), "Exit Date": df.index[i].date(), "Entry Price": entry_price, "Exit Price": exit_p, "P&L %": round(pnl, 2)})

            if trades:
                t_df = pd.DataFrame(trades)
                st.subheader("📊 Quant Scorecard")
                st.metric("Total Return", f"{t_df['P&L %'].sum():.2f}%")
                st.line_chart(t_df['P&L %'].cumsum())
                st.dataframe(t_df)