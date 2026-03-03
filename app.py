import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v4.5", layout="wide")

st.title("🛡️ Pro-Tracer: RSI Power-Level Suite (40/50/60)")
st.sidebar.header("Scan Parameters")

ticker = st.sidebar.text_input("Ticker Symbol", "TRENT.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(1999, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

mode = st.sidebar.radio("Module", ["Global 3D Optimizer", "Trade Detailer"])

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
        st.header("🔬 High-Precision 3D Scan (40/50/60 Matrix)")
        st.info("Scanning 125 Lengths (Step 2) against Entry/Exit combos of 40, 50, and 60.")

        if st.button("🚀 Start Power-Level Scan"):
            all_results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            
            # --- PARAMETER SPACE ---
            len_range = range(3, 253, 2) 
            # Limited strictly to 40, 50, 60
            power_levels = [40, 50, 60] 
            
            progress_bar = st.progress(0)
            status = st.empty()
            
            # Optimization: Pre-calculate Trend Filter
            df['Trend_OK'] = df['Close'] > ta.ema(df['Close'], length=200)

            total_len = len(len_range)
            for idx, r_len in enumerate(len_range):
                rsi = ta.rsi(df['Close'], length=r_len)
                if rsi is None: continue
                
                for ent in power_levels:
                    for ext in power_levels:
                        if ext >= ent: continue # Logic: Exit must be lower than Entry
                        
                        # Simulation logic
                        sig = pd.Series(0, index=df.index)
                        in_pos = False
                        for j in range(1, len(df)):
                            if not in_pos and rsi.iloc[j] > ent and df['Trend_OK'].iloc[j]: in_pos = True
                            elif in_pos and rsi.iloc[j] < ext: in_pos = False
                            sig.iloc[j] = 1 if in_pos else 0
                        
                        strat_ret = (df['Market_Ret'] * sig.shift(1)).fillna(0)
                        cum_ret = (1 + strat_ret).cumprod()
                        roi = (cum_ret.iloc[-1] - 1) * 100
                        max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min() * 100
                        
                        all_results.append({
                            'RSI_Len': r_len, 'Entry': ent, 'Exit': ext, 
                            'ROI %': round(roi, 2), 'Max_DD %': round(max_dd, 2),
                            'Recov_Factor': round(abs(roi/max_dd), 2) if max_dd != 0 else 0
                        })
                progress_bar.progress((idx + 1) / total_len)

            res_df = pd.DataFrame(all_results)
            
            # --- HEATMAP: RECOVERY FACTOR (RISK-ADJUSTED) ---
            st.write("### 🌡️ Risk-Adjusted Heatmap (Recovery Factor)")
            st.caption("This shows which RSI lengths offer the smoothest ride (Net Profit vs Max Drawdown).")
            
            # Grouping by Length to see structural stability
            heat_data = res_df.groupby('RSI_Len')['Recov_Factor'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(15, 2.5))
            sns.heatmap(heat_data.set_index('RSI_Len').T, cmap="YlGn", cbar_kws={'label': 'Avg Recov Factor'}, ax=ax)
            ax.set_title("Strategy Stability Map (3-252)")
            st.pyplot(fig)

            # --- TOP RESULTS ---
            st.subheader("🏆 Top Ranked Strategies (40/50/60 Matrix)")
            st.dataframe(res_df.sort_values('ROI %', ascending=False).head(15), use_container_width=True)

    # --- MODULE 2: TRADE DETAILER ---
    elif mode == "Trade Detailer":
        st.header("📜 Performance Audit")
        c1, c2, c3, c4 = st.columns(4)
        in_rsi = c1.number_input("RSI Length", value=14)
        in_ent = c2.selectbox("Entry RSI", [40, 50, 60], index=2)
        in_ext = c3.selectbox("Exit RSI", [40, 50, 60], index=1)
        vol_mult = c4.number_input("Vol Spike (x)", value=1.5)
        
        if st.button("📊 Run Detailed Audit"):
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
                vol_spike = (df['Volume'].iloc[i] > (df['Vol_MA'].iloc[i] * vol_mult)) if not pd.isna(df['Vol_MA'].iloc[i]) else False
                
                if not in_trade:
                    if rsi_v > in_ent and prev_rsi <= in_ent and curr_p > df['Trend_EMA'].iloc[i] and vol_spike:
                        in_trade, entry_date, entry_price = True, df.index[i], curr_p
                elif in_trade:
                    if rsi_v < in_ext and prev_rsi >= in_ext:
                        in_trade = False
                        exit_p = curr_p
                        pnl = ((exit_p - entry_price) / entry_price) * 100
                        trades.append({"Entry Date": entry_date.date(), "Exit Date": df.index[i].date(), "Entry Price": round(entry_price, 2), "Exit Price": round(exit_p, 2), "P&L %": round(pnl, 2)})

            if trades:
                t_df = pd.DataFrame(trades)
                daily_rets = pd.Series(0.0, index=df.index)
                for _, row in t_df.iterrows(): daily_rets.loc[pd.to_datetime(row['Exit Date'])] = row['P&L %'] / 100
                cum_strategy = (1 + daily_rets).cumprod()
                
                max_dd_val = ((cum_strategy - cum_strategy.cummax()) / cum_strategy.cummax()).min()
                
                s1, s2, s3 = st.columns(3)
                s1.metric("Total ROI", f"{(cum_strategy.iloc[-1]-1)*100:.2f}%")
                s2.metric("Max Drawdown", f"{max_dd_val*100:.2f}%")
                s3.metric("Num Trades", len(t_df))
                
                st.line_chart(100000 * cum_strategy)
                st.dataframe(t_df, use_container_width=True)