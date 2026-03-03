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

# --- Data Engine ---
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
        st.info("Testing RSI (3-252) and Average Line (3-50) for maximum Risk-Adjusted Return.")
        
        if st.button("🚀 Run Full Optimization"):
            results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            
            # Step ranges to balance speed and accuracy
            rsi_range = range(3, 253, 5) 
            ema_range = range(3, 51, 5)
            
            progress_bar = st.progress(0)
            total_steps = len(rsi_range)

            for i, r_len in enumerate(rsi_range):
                progress_bar.progress(i / total_steps)
                rsi_series = ta.rsi(df['Close'], length=r_len)
                
                for e_len in ema_range:
                    rsi_ema = ta.ema(rsi_series, length=e_len)
                    
                    # Signal Calculation
                    signal = (rsi_series > rsi_ema).astype(int)
                    strat_ret = (df['Market_Ret'] * signal.shift(1)).fillna(0)
                    cum_ret = (1 + strat_ret).cumprod()
                    
                    # Metrics
                    total_return = cum_ret.iloc[-1] - 1
                    running_max = cum_ret.cummax()
                    drawdown = (cum_ret - running_max) / running_max
                    max_dd = drawdown.min()
                    risk_score = abs(total_return / max_dd) if max_dd != 0 else 0
                    
                    results.append({
                        'RSI_Len': r_len,
                        'EMA_Len': e_len,
                        'ROI %': round(total_return * 100, 2),
                        'Max_DD %': round(max_dd * 100, 2),
                        'Risk_Score': round(risk_score, 2)
                    })
            
            res_df = pd.DataFrame(results)
            st.success("Optimization Complete!")
            st.write("### Top Parameters (Sorted by ROI)")
            st.dataframe(res_df.sort_values('ROI %', ascending=False).head(20))

    # --- MODULE 2: TRADE DETAILER ---
    elif mode == "Trade Detailer":
        st.header("📜 Trade Detailer & Quant Scorecard")
        
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
                    in_trade = True
                    entry_date = df.index[i]
                    entry_price = df['Close'].iloc[i]
                elif df['Change'].iloc[i] == -1 and in_trade:
                    in_trade = False
                    exit_date = df.index[i]
                    exit_price = df['Close'].iloc[i]
                    pnl = ((exit_price - entry_price) / entry_price) * 100
                    
                    trades.append({
                        "Entry Date": entry_date.date(),
                        "Exit Date": exit_date.date(),
                        "Entry Price": round(float(entry_price), 2),
                        "Exit Price": round(float(exit_price), 2),
                        "P&L %": round(float(pnl), 2)
                    })

            if trades:
                t_df = pd.DataFrame(trades)
                
                # --- Advanced Quant Calculations ---
                gains = t_df[t_df['P&L %'] > 0]['P&L %'].sum()
                losses = abs(t_df[t_df['P&L %'] < 0]['P&L %'].sum())
                profit_factor = gains / losses if losses != 0 else np.inf
                
                daily_ret = df['Close'].pct_change() * df['Signal'].shift(1)
                sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() != 0 else 0
                
                # --- Dashboard Display ---
                st.subheader("📊 Pro-Tracer Quant Scorecard")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Win Rate", f"{(t_df['P&L %'] > 0).mean()*100:.1f}%")
                m2.metric("Profit Factor", f"{profit_factor:.2f}")
                m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                m4.metric("Avg Trade %", f"{t_df['P&L %'].mean():.2f}%")

                st.write("### P&L Distribution per Trade")
                st.bar_chart(t_df['P&L %'])
                
                st.write("### Detailed Trade Ledger")
                st.dataframe(t_df)
                
                # Equity Curve
                st.write("### Strategy Equity Curve")
                cum_strategy = (1 + daily_ret.fillna(0)).cumprod()
                st.line_chart(cum_strategy)
            else:
                st.warning("No trades found. RSI may have never crossed the EMA in this range.")