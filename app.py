import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v2.9", layout="wide")

st.title("🛡️ Pro-Tracer: RSI 60/50 Momentum Engine")
st.sidebar.header("Global Settings")

# --- Global Inputs ---
ticker = st.sidebar.text_input("Ticker Symbol", "TRENT.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(2010, 1, 1), min_value=datetime.date(1990, 1, 1))
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
    st.warning("Awaiting data... Please verify the ticker and date range.")
else:
    # --- MODULE 1: BRUTE-FORCE OPTIMIZER (Updated for Fixed RSI Zones) ---
    if mode == "Brute-Force Optimizer":
        st.header("🔍 RSI Period Optimizer")
        st.info("Finding the best RSI Look-back for the 60/50 Strategy.")
        if st.button("🚀 Run Optimization"):
            results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            
            # Testing different RSI lengths for the 60/50 logic
            rsi_range = range(5, 41, 1) 
            progress_bar = st.progress(0)
            
            for i, r_len in enumerate(rsi_range):
                progress_bar.progress((i + 1) / len(rsi_range))
                rsi = ta.rsi(df['Close'], length=r_len)
                
                # Signal Generation for Optimizer
                # Entry = RSI > 60, Exit = RSI < 50
                sig = pd.Series(0, index=df.index)
                in_pos = False
                for j in range(1, len(df)):
                    if not in_pos and rsi.iloc[j] > 60:
                        in_pos = True
                    elif in_pos and rsi.iloc[j] < 50:
                        in_pos = False
                    sig.iloc[j] = 1 if in_pos else 0
                
                strat_ret = (df['Market_Ret'] * sig.shift(1)).fillna(0)
                cum_ret = (1 + strat_ret).cumprod()
                total_return = cum_ret.iloc[-1] - 1
                max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
                
                results.append({
                    'RSI_Len': r_len,
                    'ROI %': round(total_return * 100, 2),
                    'Max_DD %': round(max_dd * 100, 2),
                    'Profit/DD': round(abs(total_return / max_dd), 2) if max_dd != 0 else 0
                })
            st.success("Optimization Complete!")
            st.dataframe(pd.DataFrame(results).sort_values('ROI %', ascending=False).head(20))

    # --- MODULE 2: TRADE DETAILER (Updated for 60/50 Logic) ---
    elif mode == "Trade Detailer":
        st.header("📜 Trade Detailer & Crash Analysis")
        col1, col2, col3 = st.columns(3)
        in_rsi = col1.number_input("RSI Look-back", value=14, min_value=3)
        stop_loss_pct = col2.number_input("Stop Loss % (0 to disable)", value=5.0, min_value=0.0)
        trend_ema_len = col3.number_input("Trend Filter (EMA Length)", value=200, min_value=0)
        
        if st.button("📊 Generate Quant Report"):
            df = df_raw.copy()
            df['RSI'] = ta.rsi(df['Close'], length=in_rsi)
            if trend_ema_len > 0:
                df['Trend_EMA'] = ta.ema(df['Close'], length=trend_ema_len)
            
            trades = []
            in_trade = False
            entry_price = 0

            for i in range(1, len(df)):
                current_price = float(df['Close'].iloc[i])
                rsi_val = df['RSI'].iloc[i]
                prev_rsi = df['RSI'].iloc[i-1]
                trend_ok = current_price > df['Trend_EMA'].iloc[i] if trend_ema_len > 0 else True

                # --- NEW ENTRY CONDITION: RSI CROSSES 60 ---
                if not in_trade:
                    if rsi_val > 60 and prev_rsi <= 60 and trend_ok:
                        in_trade, entry_date, entry_price = True, df.index[i], current_price
                
                # --- NEW EXIT CONDITION: RSI CROSSES 50 OR STOP LOSS ---
                elif in_trade:
                    sl_hit = stop_loss_pct > 0 and current_price <= entry_price * (1 - stop_loss_pct/100)
                    rsi_exit = rsi_val < 50 and prev_rsi >= 50
                    
                    if sl_hit or rsi_exit:
                        in_trade = False
                        exit_price = entry_price * (1 - stop_loss_pct/100) if sl_hit else current_price
                        reason = "Stop Loss" if sl_hit else "RSI 50 Exit"
                        pnl = ((exit_price - entry_price) / entry_price) * 100
                        trades.append({
                            "Entry Date": entry_date.date(), "Exit Date": df.index[i].date(), 
                            "Entry Price": round(entry_price, 2), "Exit Price": round(exit_price, 2), 
                            "P&L %": round(pnl, 2), "Exit Reason": reason
                        })

            if trades:
                t_df = pd.DataFrame(trades)
                
                # Drawdown Analysis
                daily_rets = pd.Series(0.0, index=df.index)
                for _, row in t_df.iterrows():
                    daily_rets.loc[pd.to_datetime(row['Exit Date'])] = row['P&L %'] / 100
                cum_strategy = (1 + daily_rets).cumprod()
                
                running_max = cum_strategy.cummax()
                drawdowns = (cum_strategy - running_max) / running_max
                max_dd_val = drawdowns.min()
                end_date_dd = drawdowns.idxmin()
                start_date_dd = cum_strategy[:end_date_dd].idxmax()
                dd_duration = (end_date_dd - start_date_dd).days

                # Scorecard Metrics
                total_ret_val = cum_strategy.iloc[-1] - 1
                num_years = max((df.index[-1] - df.index[0]).days / 365.25, 0.1)
                cagr = ((1 + total_ret_val)**(1/num_years) - 1) if total_ret_val > -1 else -1
                win_rate = (t_df['P&L %'] > 0).mean()
                profit_factor = (t_df[t_df['P&L %'] > 0]['P&L %'].sum()) / abs(t_df[t_df['P&L %'] < 0]['P&L %'].sum()) if abs(t_df[t_df['P&L %'] < 0]['P&L %'].sum()) != 0 else np.inf

                # UI Display
                st.subheader("⚠️ Max Drawdown Timing")
                d1, d2, d3 = st.columns(3)
                d1.metric("Peak Date", start_date_dd.strftime('%d %b %Y'))
                d2.metric("Bottom Date", end_date_dd.strftime('%d %b %Y'))
                d3.metric("Crash Duration", f"{dd_duration} Days")
                st.error(f"Worst Crash: {max_dd_val*100:.2f}%")

                st.write("---")
                st.subheader("📊 Quant Scorecard")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total Return", f"{total_ret_val*100:.2f}%")
                s2.metric("CAGR", f"{cagr*100:.2f}%")
                s3.metric("Max Drawdown", f"{max_dd_val*100:.2f}%")
                s4.metric("Win Rate", f"{win_rate*100:.1f}%")

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.write("### Exit Reason Breakdown")
                    exit_counts = t_df['Exit Reason'].value_counts()
                    fig, ax = plt.subplots()
                    ax.pie(exit_counts, labels=exit_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
                    st.pyplot(fig)
                with chart_col2:
                    st.write("### Compounding Chart (₹1L Start)")
                    st.line_chart(100000 * cum_strategy)
                
                st.write("### Detailed Trade Ledger")
                st.dataframe(t_df)
                
                csv = t_df.to_csv(index=False).encode('utf-8')
                st.sidebar.download_button(label="📥 Download Ledger as CSV", data=csv, file_name=f"{ticker}_backtest.csv", mime="text/csv")
            else:
                st.warning("No trades found for this RSI range.")