import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v2.0", layout="wide")

st.title("🛡️ Pro-Tracer: The Quantitative Engine")
st.sidebar.header("Global Settings")

# --- Global Inputs ---
ticker = st.sidebar.text_input("Ticker Symbol", "RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(1990, 1, 1), min_value=datetime.date(1990, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Module Navigation
mode = st.sidebar.radio("Select Module", ["Brute-Force Optimizer", "Trade Detailer"])

@st.cache_data
def get_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        if not data.empty:
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
            
            # Step 5 for speed; change to 1 for high precision
            rsi_range = range(3, 253, 5) 
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
                    
                    results.append({
                        'RSI_Len': r_len, 'EMA_Len': e_len,
                        'ROI %': round(total_return * 100, 2),
                        'Max_DD %': round(max_dd * 100, 2),
                        'Risk_Score': round(abs(total_return / max_dd), 2) if max_dd != 0 else 0
                    })
            st.success("Optimization Complete!")
            st.dataframe(pd.DataFrame(results).sort_values('ROI %', ascending=False).head(20))

    # --- MODULE 2: TRADE DETAILER ---
    elif mode == "Trade Detailer":
        st.header("📜 Trade Detailer & Advanced Scorecard")
        col1, col2 = st.columns(2)
        in_rsi = col1.number_input("RSI Look-back", value=14, min_value=3)
        in_ema = col2.number_input("EMA (Average Line) Length", value=9, min_value=3)
        
        if st.button("📊 Generate Quant Report"):
            df = df_raw.copy()
            df['RSI'] = ta.rsi(df['Close'], length=in_rsi)
            df['RSI_EMA'] = ta.ema(df['RSI'], length=in_ema)
            df['Signal'] = (df['RSI'] > df['RSI_EMA']).astype(int)
            df['Change'] = df['Signal'].diff()

            trades = []
            in_trade = False
            for i in range(1, len(df)):
                if df['Change'].iloc[i] == 1 and not in_trade:
                    in_trade, entry_date, entry_price = True, df.index[i], df['Close'].iloc[i]
                elif df['Change'].iloc[i] == -1 and in_trade:
                    in_trade, exit_date, exit_price = False, df.index[i], df['Close'].iloc[i]
                    pnl = ((exit_price - entry_price) / entry_price) * 100
                    trades.append({"Entry Date": entry_date.date(), "Exit Date": exit_date.date(), 
                                   "Entry Price": float(entry_price), "Exit Price": float(exit_price), "P&L %": float(pnl)})

            if trades:
                t_df = pd.DataFrame(trades)
                
                # --- Advanced Calculations ---
                daily_ret = (df['Close'].pct_change() * df['Signal'].shift(1)).fillna(0)
                cum_strategy = (1 + daily_ret).cumprod()
                
                total_ret = (cum_strategy.iloc[-1] - 1)
                num_years = (df.index[-1] - df.index[0]).days / 365.25
                cagr = ((1 + total_ret)**(1/num_years) - 1) if total_ret > -1 else -1
                max_dd = ((cum_strategy - cum_strategy.cummax()) / cum_strategy.cummax()).min()
                
                win_rate = (t_df['P&L %'] > 0).mean()
                avg_win = t_df[t_df['P&L %'] > 0]['P&L %'].mean()
                avg_loss = abs(t_df[t_df['P&L %'] < 0]['P&L %'].mean())
                expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
                profit_factor = (t_df[t_df['P&L %'] > 0]['P&L %'].sum()) / abs(t_df[t_df['P&L %'] < 0]['P&L %'].sum()) if losses != 0 else np.inf

                # --- 1. Quant Scorecard ---
                st.subheader("📊 Pro-Tracer Quant Scorecard")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total Return", f"{total_ret*100:.2f}%")
                s2.metric("CAGR", f"{cagr*100:.2f}%")
                s3.metric("Max Drawdown", f"{max_dd*100:.2f}%")
                s4.metric("Expectancy", f"{expectancy:.2f}%")
                
                s5, s6, s7, s8 = st.columns(4)
                s5.metric("Win Rate", f"{win_rate*100:.1f}%")
                s6.metric("Profit Factor", f"{profit_factor:.2f}")
                s7.metric("Avg Trade %", f"{t_df['P&L %'].mean():.2f}%")
                s8.metric("Num Trades", len(t_df))

                # --- 2. Compounding Calculator ---
                st.subheader("💰 The Power of Compounding")
                initial_capital = 100000
                final_value = initial_capital * (1 + total_ret)
                st.write(f"If you started with **₹1,00,000**, it would now be worth: **₹{final_value:,.2f}**")
                
                comp_curve = initial_capital * cum_strategy
                st.line_chart(comp_curve)

                st.write("### Detailed Trade Ledger")
                st.dataframe(t_df)
            else:
                st.warning("No trades found.")