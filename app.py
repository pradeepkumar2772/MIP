import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v3.6", layout="wide")

st.title("🛡️ Pro-Tracer: Full-Spectrum Optimizer")
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
        st.header("📈 Full RSI Sensitivity Analysis (3 to 252)")
        
        if st.button("🚀 Run Full Spectrum Scan"):
            results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            
            # Range 3-252, Step 5 (Total 50 test cases)
            rsi_range = range(3, 253, 5) 
            progress_bar = st.progress(0)
            
            for i, r_len in enumerate(rsi_range):
                progress_bar.progress((i + 1) / len(rsi_range))
                rsi = ta.rsi(df['Close'], length=r_len)
                
                # Simulation Logic (Entry 60 / Exit 50)
                sig = pd.Series(0, index=df.index)
                in_pos = False
                for j in range(1, len(df)):
                    if not in_pos and rsi.iloc[j] > 60: in_pos = True
                    elif in_pos and rsi.iloc[j] < 50: in_pos = False
                    sig.iloc[j] = 1 if in_pos else 0
                
                strat_ret = (df['Market_Ret'] * sig.shift(1)).fillna(0)
                cum_ret = (1 + strat_ret).cumprod()
                total_return = (cum_ret.iloc[-1] - 1) * 100
                max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min() * 100
                
                results.append({
                    'RSI_Len': r_len, 
                    'ROI %': round(total_return, 2),
                    'Max_DD %': round(max_dd, 2),
                    'Recovery Factor': round(abs(total_return / max_dd), 2) if max_dd != 0 else 0
                })
            
            res_df = pd.DataFrame(results)
            
            # 1. Visualization
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(res_df['RSI_Len'], res_df['ROI %'], marker='o', color='#2ca02c', label='ROI %')
            ax.set_xlabel("RSI Look-back Period")
            ax.set_ylabel("Total ROI %")
            ax.set_title(f"Full Strategy Sensitivity: {ticker}")
            ax.grid(True, alpha=0.2)
            st.pyplot(fig)
            
            # 2. Complete Ranked List (No .head() restriction)
            st.write("### 🏆 Full Ranked Strategy List")
            st.info("Showing all 50+ combinations tested. Sort by clicking column headers.")
            
            # Highlight high Recovery Factors
            full_list = res_df.sort_values('ROI %', ascending=False).reset_index(drop=True)
            st.dataframe(full_list, use_container_width=True)
            
            # 3. Data Export
            csv_opt = full_list.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download All Optimizer Results", data=csv_opt, file_name=f"{ticker}_full_optimization.csv", mime="text/csv")

    elif mode == "Trade Detailer":
        # Trade Detailer logic remains stable with v3.3 features (Prices, Vol Spike, Scorecard)
        st.header("📜 Trade Detailer & Recovery Analysis")
        col1, col2, col3, col4 = st.columns(4)
        in_rsi = col1.number_input("RSI Look-back", value=14)
        vol_mult = col2.number_input("Vol Spike (x Avg)", value=1.5)
        vol_ma_p = col3.number_input("Vol Avg Period", value=20)
        stop_loss_p = col4.number_input("Stop Loss %", value=5.0)
        
        if st.button("📊 Generate Detailed Report"):
            df = df_raw.copy()
            df['RSI'] = ta.rsi(df['Close'], length=in_rsi)
            df['Vol_MA'] = df['Volume'].rolling(window=vol_ma_p).mean()
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
                    sl_hit = stop_loss_p > 0 and curr_p <= entry_price * (1 - stop_loss_p/100)
                    rsi_exit = rsi_v < 50 and prev_rsi >= 50
                    if sl_hit or rsi_exit:
                        in_trade = False
                        exit_p = entry_price * (1 - stop_loss_p/100) if sl_hit else curr_p
                        pnl = ((exit_p - entry_price) / entry_price) * 100
                        trades.append({
                            "Entry Date": entry_date.date(), "Exit Date": df.index[i].date(), 
                            "Entry Price": round(entry_price, 2), "Exit Price": round(exit_p, 2), 
                            "P&L %": round(pnl, 2), "Reason": "SL" if sl_hit else "RSI 50"
                        })

            if trades:
                t_df = pd.DataFrame(trades)
                # Performance Visuals
                daily_rets = pd.Series(0.0, index=df.index)
                for _, row in t_df.iterrows(): daily_rets.loc[pd.to_datetime(row['Exit Date'])] = row['P&L %'] / 100
                cum_strategy = (1 + daily_rets).cumprod()
                
                st.subheader("📊 Quant Scorecard")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total ROI", f"{(cum_strategy.iloc[-1]-1)*100:.2f}%")
                c2.metric("Win Rate", f"{(t_df['P&L %']>0).mean()*100:.1f}%")
                c3.metric("Max DD", f"{((cum_strategy - cum_strategy.cummax()) / cum_strategy.cummax()).min()*100:.2f}%")
                
                st.line_chart(100000 * cum_strategy)
                st.dataframe(t_df, use_container_width=True)
            else:
                st.warning("No trades found.")