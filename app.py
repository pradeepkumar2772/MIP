import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v5.0", layout="wide")

st.title("🛡️ Pro-Tracer: Consistency & Profit-Factor Suite")
st.sidebar.header("Global Settings")

ticker = st.sidebar.text_input("Ticker Symbol", "TRENT.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(1999, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

mode = st.sidebar.radio("Select Module", ["Global 3D Optimizer", "Trade Detailer"])

@st.cache_data
def get_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    if not data.empty and isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

df_raw = get_data(ticker, start_date, end_date)

if df_raw.empty:
    st.warning("Awaiting data... Please check the ticker symbol.")
else:
    # --- MODULE 1: GLOBAL 3D OPTIMIZER ---
    if mode == "Global 3D Optimizer":
        st.header("🔬 Narrow-Band Strategy Optimizer")
        st.info("Scanning RSI Lengths (3-252) | Entry (50-60) | Exit (40-60)")

        if st.button("🚀 Start Precision Scan"):
            all_results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            
            len_range = range(3, 253, 10) 
            ent_range = range(50, 61, 2)  
            ext_range = range(40, 61, 2)  
            
            progress_bar = st.progress(0)
            total_combos = len(len_range) * len(ent_range) * len(ext_range)
            count = 0

            for r_len in len_range:
                rsi = ta.rsi(df['Close'], length=r_len)
                if rsi is None: continue
                
                for ent in ent_range:
                    for ext in ext_range:
                        if ext >= ent: continue 
                        
                        count += 1
                        progress_bar.progress(count / total_combos)

                        # Logic Simulation
                        in_pos = False
                        trade_results = []
                        entry_p = 0
                        
                        for j in range(1, len(df)):
                            val = rsi.iloc[j]
                            curr_c = float(df['Close'].iloc[j])
                            if not in_pos and val > ent:
                                in_pos, entry_p = True, curr_c
                            elif in_pos and val < ext:
                                in_pos = False
                                trade_results.append((curr_c - entry_p) / entry_p)

                        if trade_results:
                            roi = (np.prod([1 + r for r in trade_results]) - 1) * 100
                            # Win Rate Calculation
                            wins = [r for r in trade_results if r > 0]
                            losses = [r for r in trade_results if r <= 0]
                            win_rate = (len(wins) / len(trade_results)) * 100
                            
                            # Profit Factor Calculation
                            gross_profit = sum(wins)
                            gross_loss = abs(sum(losses))
                            profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
                            
                            all_results.append({
                                'RSI_Len': r_len, 'Entry': ent, 'Exit': ext, 
                                'ROI %': round(roi, 2), 
                                'Win Rate %': round(win_rate, 1),
                                'Profit Factor': round(profit_factor, 2),
                                'Trades': len(trade_results)
                            })

            res_df = pd.DataFrame(all_results).sort_values('ROI %', ascending=False)
            st.success("Scan Complete!")
            st.dataframe(res_df, use_container_width=True)

    # --- MODULE 2: TRADE DETAILER ---
    elif mode == "Trade Detailer":
        st.header("📜 Performance Audit & Ledger")
        c1, c2, c3, c4 = st.columns(4)
        in_rsi = c1.number_input("RSI Length", value=14, min_value=3)
        in_ent = c2.slider("Entry Threshold", 50, 60, 55)
        in_ext = c3.slider("Exit Threshold", 40, 60, 45)
        vol_mult = c4.number_input("Vol Spike (x)", value=1.5, step=0.1)
        
        if st.button("📊 Generate Detailed Report"):
            df = df_raw.copy()
            df['RSI'] = ta.rsi(df['Close'], length=in_rsi)
            df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
            df['Trend_EMA'] = ta.ema(df['Close'], length=200)
            
            trades = []
            in_trade = False
            for i in range(1, len(df)):
                curr_p = float(df['Close'].iloc[i])
                rsi_v = df['RSI'].iloc[i]
                prev_rsi = df['RSI'].iloc[i-1]
                vol_spike = (df['Volume'].iloc[i] > (df['Vol_MA'].iloc[i] * vol_mult)) if not pd.isna(df['Vol_MA'].iloc[i]) else True
                
                if not in_trade:
                    if rsi_v > in_ent and prev_rsi <= in_ent and curr_p > df['Trend_EMA'].iloc[i] and vol_spike:
                        in_trade, entry_date, entry_price = True, df.index[i], curr_p
                elif in_trade:
                    if rsi_v < in_ext and prev_rsi >= in_ext:
                        in_trade = False
                        exit_p = curr_p
                        pnl = ((exit_p - entry_price) / entry_price) * 100
                        trades.append({
                            "Entry Date": entry_date.date(), "Exit Date": df.index[i].date(), 
                            "Entry Price": round(entry_price, 2), "Exit Price": round(exit_p, 2), 
                            "P&L %": round(pnl, 2)
                        })

            if trades:
                t_df = pd.DataFrame(trades)
                
                # Metrics Calculation
                wins = t_df[t_df['P&L %'] > 0]['P&L %']
                losses = t_df[t_df['P&L %'] <= 0]['P&L %']
                win_rate = (len(wins) / len(t_df)) * 100
                profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty else np.inf
                
                daily_rets = pd.Series(0.0, index=df.index)
                for _, row in t_df.iterrows():
                    daily_rets.loc[pd.to_datetime(row['Exit Date'])] = row['P&L %'] / 100
                cum_strategy = (1 + daily_rets).cumprod()
                max_dd_val = ((cum_strategy - cum_strategy.cummax()) / cum_strategy.cummax()).min()
                total_ret_val = (cum_strategy.iloc[-1] - 1) * 100
                
                st.subheader("📊 Quant Scorecard")
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric("Total ROI", f"{total_ret_val:.1f}%")
                s2.metric("Win Rate", f"{win_rate:.1f}%")
                s3.metric("Profit Factor", f"{profit_factor:.2f}")
                s4.metric("Max Drawdown", f"{max_dd_val*100:.2f}%")
                s5.metric("Recovery Factor", f"{abs(total_ret_val/(max_dd_val*100)):.2f}")
                
                st.line_chart(100000 * cum_strategy)
                st.write("### 📋 Trade Ledger")
                st.dataframe(t_df, use_container_width=True)
            else:
                st.warning("No trades found. Check if the stock is above its 200 EMA.")