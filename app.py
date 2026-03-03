import yfinance as yf
import pd as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v4.7", layout="wide")

st.title("🛡️ Pro-Tracer: Narrow-Band 3D Optimizer")
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
        st.header("🔬 Narrow-Band Strategy Optimizer")
        st.info("Scanning RSI Lengths (3-252) with Entries (50-60) and Exits (40-60).")

        if st.button("🚀 Start Precision Scan"):
            all_results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            
            # Optimization Parameters
            len_range = range(3, 253, 10) # 10-step jump for broad cycle coverage
            ent_range = range(50, 61, 2)  # High precision in narrow band
            ext_range = range(40, 61, 2)  # High precision in narrow band
            
            progress_bar = st.progress(0)
            total_combos = len(len_range) * len(ent_range) * len(ext_range)
            count = 0

            for r_len in len_range:
                rsi = ta.rsi(df['Close'], length=r_len)
                for ent in ent_range:
                    for ext in ext_range:
                        if ext >= ent: continue # Logical check: Must exit lower than entry
                        
                        count += 1
                        progress_bar.progress(count / total_combos)

                        sig = pd.Series(0, index=df.index)
                        in_pos = False
                        for j in range(1, len(df)):
                            if not in_pos and rsi.iloc[j] > ent: in_pos = True
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

            res_df = pd.DataFrame(all_results).sort_values('ROI %', ascending=False)
            st.success("Precision Scan Complete!")
            st.dataframe(res_df, use_container_width=True)

    # --- MODULE 2: TRADE DETAILER ---
    elif mode == "Trade Detailer":
        st.header("📜 Performance Audit")
        c1, c2, c3, c4 = st.columns(4)
        in_rsi = c1.number_input("RSI Length", value=14)
        in_ent = c2.slider("Entry RSI", 50, 60, 55)
        in_ext = c3.slider("Exit RSI", 40, 60, 45)
        vol_mult = c4.number_input("Vol Spike (x)", value=1.5)
        
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
                vol_spike = (df['Volume'].iloc[i] > (df['Vol_MA'].iloc[i] * vol_mult)) if not pd.isna(df['Vol_MA'].iloc[i]) else False
                
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
                daily_rets = pd.Series(0.0, index=df.index)
                for _, row in t_df.iterrows():
                    daily_rets.loc[pd.to_datetime(row['Exit Date'])] = row['P&L %'] / 100
                cum_strategy = (1 + daily_rets).cumprod()
                
                max_dd_val = ((cum_strategy - cum_strategy.cummax()) / cum_strategy.cummax()).min()
                total_ret_val = cum_strategy.iloc[-1] - 1
                rec_factor = abs(total_ret_val / max_dd_val) if max_dd_val != 0 else np.inf
                
                st.subheader("📊 Quant Scorecard")
                s1, s2, s3 = st.columns(3)
                s1.metric("Total ROI", f"{total_ret_val*100:.1f}%")
                s2.metric("Max Drawdown", f"{max_dd_val*100:.2f}%")
                s3.metric("Recovery Factor", f"{rec_factor:.2f}")
                
                st.line_chart(100000 * cum_strategy)
                st.dataframe(t_df, use_container_width=True)