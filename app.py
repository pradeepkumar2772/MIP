import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v3.0", layout="wide")

st.title("🛡️ Pro-Tracer: Institutional Momentum (RSI 60 + Vol)")
st.sidebar.header("Global Settings")

# --- Global Inputs ---
ticker = st.sidebar.text_input("Ticker Symbol", "TRENT.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(1999, 1, 1))
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
    # --- MODULE 1: OPTIMIZER ---
    if mode == "Brute-Force Optimizer":
        st.header("🔍 RSI Period Optimizer")
        if st.button("🚀 Run Optimization"):
            results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            rsi_range = range(5, 41, 2) 
            progress_bar = st.progress(0)
            
            for i, r_len in enumerate(rsi_range):
                progress_bar.progress((i + 1) / len(rsi_range))
                rsi = ta.rsi(df['Close'], length=r_len)
                sig = pd.Series(0, index=df.index)
                in_pos = False
                for j in range(1, len(df)):
                    if not in_pos and rsi.iloc[j] > 60: in_pos = True
                    elif in_pos and rsi.iloc[j] < 50: in_pos = False
                    sig.iloc[j] = 1 if in_pos else 0
                
                strat_ret = (df['Market_Ret'] * sig.shift(1)).fillna(0)
                cum_ret = (1 + strat_ret).cumprod()
                total_return = cum_ret.iloc[-1] - 1
                max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
                results.append({'RSI_Len': r_len, 'ROI %': round(total_return * 100, 2), 'Max_DD %': round(max_dd * 100, 2)})
            st.dataframe(pd.DataFrame(results).sort_values('ROI %', ascending=False).head(20))

    # --- MODULE 2: TRADE DETAILER ---
    elif mode == "Trade Detailer":
        st.header("📜 Institutional Detailer & Recovery Analysis")
        col1, col2, col3, col4 = st.columns(4)
        in_rsi = col1.number_input("RSI Look-back", value=14)
        vol_mult = col2.number_input("Vol Spike (x Avg)", value=1.5, step=0.1)
        vol_ma = col3.number_input("Vol Avg Period", value=20)
        stop_loss_pct = col4.number_input("Stop Loss %", value=5.0)
        
        if st.button("📊 Generate Institutional Report"):
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
                vol_v = df['Volume'].iloc[i]
                v_ma = df['Vol_MA'].iloc[i]
                
                # --- ENTRY: RSI > 60 + VOL SPIKE + ABOVE 200 EMA ---
                if not in_trade:
                    vol_spike = vol_v > (v_ma * vol_mult)
                    if rsi_v > 60 and prev_rsi <= 60 and curr_p > df['Trend_EMA'].iloc[i] and vol_spike:
                        in_trade, entry_date, entry_price = True, df.index[i], curr_p
                
                # --- EXIT: RSI < 50 OR STOP LOSS ---
                elif in_trade:
                    sl_hit = stop_loss_pct > 0 and curr_p <= entry_price * (1 - stop_loss_pct/100)
                    rsi_exit = rsi_v < 50 and prev_rsi >= 50
                    if sl_hit or rsi_exit:
                        in_trade = False
                        exit_p = entry_price * (1 - stop_loss_pct/100) if sl_hit else curr_p
                        reason = "Stop Loss" if sl_hit else "RSI 50 Exit"
                        pnl = ((exit_p - entry_price) / entry_price) * 100
                        trades.append({"Entry Date": entry_date.date(), "Exit Date": df.index[i].date(), "P&L %": round(pnl, 2), "Exit Reason": reason})

            if trades:
                t_df = pd.DataFrame(trades)
                
                # Performance & Recovery Factor
                daily_rets = pd.Series(0.0, index=df.index)
                for _, row in t_df.iterrows():
                    daily_rets.loc[pd.to_datetime(row['Exit Date'])] = row['P&L %'] / 100
                cum_strategy = (1 + daily_rets).cumprod()
                
                max_dd_val = ((cum_strategy - cum_strategy.cummax()) / cum_strategy.cummax()).min()
                total_ret_val = cum_strategy.iloc[-1] - 1
                
                # Recovery Factor = Net Profit / Max Drawdown
                recovery_factor = abs(total_ret_val / max_dd_val) if max_dd_val != 0 else np.inf
                
                st.subheader("📊 Institutional Scorecard")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total ROI", f"{total_ret_val*100:.1f}%")
                s2.metric("Max Drawdown", f"{max_dd_val*100:.2f}%")
                s3.metric("Recovery Factor", f"{recovery_factor:.2f}", help="Net Profit / Max Drawdown")
                s4.metric("Win Rate", f"{(t_df['P&L %'] > 0).mean()*100:.1f}%")

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.write("### Exit Breakdown")
                    st.pyplot(plt.subplots()[0].pie(t_df['Exit Reason'].value_counts(), labels=t_df['Exit Reason'].value_counts().index, autopct='%1.1f%%')[0].get_figure())
                with chart_col2:
                    st.write("### Equity Curve")
                    st.line_chart(100000 * cum_strategy)
                
                st.dataframe(t_df)
            else:
                st.warning("No institutional setups found. Try lowering the Volume Spike multiplier.")