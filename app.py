import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v2.7", layout="wide")

st.title("🛡️ Pro-Tracer: Advanced Momentum & Trend Engine")
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
    st.warning("Awaiting data... Please verify the ticker and date range.")
else:
    # --- MODULE 1: BRUTE-FORCE OPTIMIZER ---
    if mode == "Brute-Force Optimizer":
        st.header("🔍 Brute-Force Parameter Optimizer")
        if st.button("🚀 Run Full Optimization"):
            results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            rsi_range = range(3, 253, 10) 
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

    # --- MODULE 2: TRADE DETAILER ---
    elif mode == "Trade Detailer":
        st.header("📜 Trade Detailer & Advanced Analysis")
        # Added col5 for Entry RSI Min
        col1, col2, col3, col4, col5 = st.columns(5)
        in_rsi = col1.number_input("RSI Look-back", value=14, min_value=3)
        in_ema = col2.number_input("Signal EMA Length", value=9, min_value=3)
        stop_loss_pct = col3.number_input("Stop Loss %", value=5.0, min_value=0.0, step=0.5)
        trend_ema_len = col4.number_input("Trend Filter (EMA)", value=200, min_value=0)
        rsi_min_val = col5.number_input("Entry RSI Min", value=50, min_value=0, max_value=100)
        
        if st.button("📊 Generate Quant Report"):
            df = df_raw.copy()
            df['RSI'] = ta.rsi(df['Close'], length=in_rsi)
            df['RSI_EMA'] = ta.ema(df['RSI'], length=in_ema)
            if trend_ema_len > 0:
                df['Trend_EMA'] = ta.ema(df['Close'], length=trend_ema_len)
            
            trades = []
            in_trade = False
            entry_price = 0
            for i in range(1, len(df)):
                current_price = float(df['Close'].iloc[i])
                rsi_val = df['RSI'].iloc[i]
                ema_val = df['RSI_EMA'].iloc[i]
                
                # Filter Logic
                trend_ok = current_price > df['Trend_EMA'].iloc[i] if trend_ema_len > 0 else True
                rsi_strength_ok = rsi_val >= rsi_min_val

                if not in_trade:
                    # New Entry Condition: RSI Cross UP + Trend OK + RSI above Minimum
                    if rsi_val > ema_val and df['RSI'].iloc[i-1] <= df['RSI_EMA'].iloc[i-1] and trend_ok and rsi_strength_ok:
                        in_trade, entry_date, entry_price = True, df.index[i], current_price
                elif in_trade:
                    sl_hit = stop_loss_pct > 0 and current_price <= entry_price * (1 - stop_loss_pct/100)
                    rsi_exit = rsi_val < ema_val
                    if sl_hit or rsi_exit:
                        in_trade = False
                        exit_price = entry_price * (1 - stop_loss_pct/100) if sl_hit else current_price
                        reason = "Stop Loss" if sl_hit else "RSI Crossover"
                        pnl = ((exit_price - entry_price) / entry_price) * 100
                        trades.append({
                            "Entry Date": entry_date.date(), "Exit Date": df.index[i].date(), 
                            "Entry Price": round(entry_price, 2), "Exit Price": round(exit_price, 2), 
                            "P&L %": round(pnl, 2), "Exit Reason": reason
                        })

            if trades:
                t_df = pd.DataFrame(trades)
                
                # --- DRAWDOWN ANALYSIS ---
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

                # --- Scorecard Metrics ---
                total_ret_val = cum_strategy.iloc[-1] - 1
                num_years = max((df.index[-1] - df.index[0]).days / 365.25, 0.1)
                cagr = ((1 + total_ret_val)**(1/num_years) - 1) if total_ret_val > -1 else -1
                win_rate = (t_df['P&L %'] > 0).mean()
                avg_win = t_df[t_df['P&L %'] > 0]['P&L %'].mean() if not t_df[t_df['P&L %'] > 0].empty else 0
                avg_loss = abs(t_df[t_df['P&L %'] < 0]['P&L %'].mean()) if not t_df[t_df['P&L %'] < 0].empty else 0
                expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
                profit_factor = (t_df[t_df['P&L %'] > 0]['P&L %'].sum()) / abs(t_df[t_df['P&L %'] < 0]['P&L %'].sum()) if abs(t_df[t_df['P&L %'] < 0]['P&L %'].sum()) != 0 else np.inf
                t_df['IsWin'] = t_df['P&L %'] > 0
                runs = t_df['IsWin'].groupby((t_df['IsWin'] != t_df['IsWin'].shift()).cumsum()).cumcount() + 1
                max_cons_losses = runs[t_df['IsWin'] == False].max() if not t_df[t_df['IsWin'] == False].empty else 0

                # --- UI Display ---
                st.subheader("⚠️ Max Drawdown Timing (The Worst Crash)")
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
                s4.metric("Expectancy", f"{expectancy:.2f}%")
                
                s5, s6, s7, s8 = st.columns(4)
                s5.metric("Win Rate", f"{win_rate*100:.1f}%")
                s6.metric("Profit Factor", f"{profit_factor:.2f}")
                s7.metric("Max Cons. Losses", f"{int(max_cons_losses)}")
                s8.metric("Num Trades", len(t_df))

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.write("### Exit Reason Breakdown")
                    exit_counts = t_df['Exit Reason'].value_counts()
                    fig, ax = plt.subplots()
                    ax.pie(exit_counts, labels=exit_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
                    st.pyplot(fig)
                with chart_col2:
                    st.write("### P&L Distribution")
                    st.bar_chart(t_df['P&L %'])

                st.subheader("💰 Compounding Chart (Starting ₹1,00,000)")
                st.line_chart(100000 * cum_strategy)
                st.write("### Detailed Trade Ledger")
                st.dataframe(t_df)
            else:
                st.warning("No trades found.")