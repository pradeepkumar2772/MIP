import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v5.1", layout="wide")

st.title("🛡️ Pro-Tracer: Risk-Reward & Profitability Engine")
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
    st.warning("Awaiting data...")
else:
    # --- MODULE 1: GLOBAL 3D OPTIMIZER ---
    if mode == "Global 3D Optimizer":
        st.header("🔬 Strategic Risk-Reward Optimizer")
        if st.button("🚀 Run Deep Analysis"):
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

                        in_pos, trade_results = False, []
                        entry_p = 0
                        
                        for j in range(1, len(df)):
                            val, curr_c = rsi.iloc[j], float(df['Close'].iloc[j])
                            if not in_pos and val > ent:
                                in_pos, entry_p = True, curr_c
                            elif in_pos and val < ext:
                                in_pos = False
                                trade_results.append((curr_c - entry_p) / entry_p)

                        if trade_results:
                            roi = (np.prod([1 + r for r in trade_results]) - 1) * 100
                            wins = [r for r in trade_results if r > 0]
                            losses = [r for r in trade_results if r <= 0]
                            
                            win_rate = (len(wins) / len(trade_results)) * 100
                            avg_win = np.mean(wins) if wins else 0
                            avg_loss = abs(np.mean(losses)) if losses else 0
                            rrr = avg_win / avg_loss if avg_loss != 0 else np.inf
                            
                            all_results.append({
                                'RSI_Len': r_len, 'Entry': ent, 'Exit': ext, 
                                'ROI %': round(roi, 2), 
                                'Win Rate %': round(win_rate, 1),
                                'Avg RRR': round(rrr, 2),
                                'Trades': len(trade_results)
                            })

            res_df = pd.DataFrame(all_results).sort_values('ROI %', ascending=False)
            st.success("Analysis Complete!")
            st.dataframe(res_df, use_container_width=True)

    # --- MODULE 2: TRADE DETAILER ---
    elif mode == "Trade Detailer":
        st.header("📜 Performance Audit")
        c1, c2, c3, c4 = st.columns(4)
        in_rsi = c1.number_input("RSI Length", value=14, min_value=3)
        in_ent = c2.slider("Entry Level", 50, 60, 55)
        in_ext = c3.slider("Exit Level", 40, 60, 45)
        vol_mult = c4.number_input("Vol Spike (x)", value=1.5)
        
        if st.button("📊 Generate Report"):
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
                        pnl = ((curr_p - entry_price) / entry_price) * 100
                        trades.append({"Entry": entry_date.date(), "Exit": df.index[i].date(), "Entry P": round(entry_price, 2), "Exit P": round(curr_p, 2), "P&L %": round(pnl, 2)})

            if trades:
                t_df = pd.DataFrame(trades)
                wins = t_df[t_df['P&L %'] > 0]['P&L %']
                losses = t_df[t_df['P&L %'] <= 0]['P&L %']
                
                avg_win, avg_loss = wins.mean(), abs(losses.mean())
                rrr = avg_win / avg_loss if avg_loss != 0 else np.inf
                
                daily_rets = pd.Series(0.0, index=df.index)
                for _, row in t_df.iterrows(): daily_rets.loc[pd.to_datetime(row['Exit'])] = row['P&L %'] / 100
                cum_strategy = (1 + daily_rets).cumprod()
                
                st.subheader("📊 Quant Scorecard")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total ROI", f"{(cum_strategy.iloc[-1]-1)*100:.1f}%")
                s2.metric("Win Rate", f"{(len(wins)/len(t_df))*100:.1f}%")
                s3.metric("Avg RRR", f"{rrr:.2f}:1")
                s4.metric("Trades", len(t_df))
                
                st.line_chart(100000 * cum_strategy)
                st.dataframe(t_df, use_container_width=True)