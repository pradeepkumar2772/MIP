import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v4.3", layout="wide")

st.title("🛡️ Pro-Tracer: 3D High-Precision Global Suite")
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
        st.header("🔬 High-Precision 3D Scan")
        st.info("Scanning 125 Lengths (Step 2) across 36 Entry/Exit Matrix combinations.")

        if st.button("🚀 Start High-Precision Global Scan"):
            all_results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            
            # --- HIGH PRECISION 3D SPACE ---
            len_range = range(3, 253, 2)  # Step 2 as requested
            ent_range = range(55, 81, 5)   # Entry levels
            ext_range = range(35, 61, 5)   # Exit levels
            
            total_combos = len(len_range) * len(ent_range) * len(ext_range)
            progress_bar = st.progress(0)
            status = st.empty()
            count = 0

            # Pre-calculate Trend Filter to save time in loop
            df['Trend_OK'] = df['Close'] > ta.ema(df['Close'], length=200)

            for r_len in len_range:
                # Pre-calculate RSI for this length
                rsi = ta.rsi(df['Close'], length=r_len)
                if rsi is None: continue
                
                for ent in ent_range:
                    for ext in ext_range:
                        if ext >= ent: continue 
                        
                        count += 1
                        if count % 50 == 0: # UI Refresh every 50 combos
                            progress_bar.progress(count / total_combos)
                            status.text(f"Scanning Precision Node {count}/{total_combos} | Current RSI Len: {r_len}")

                        # Simulation
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

            res_df = pd.DataFrame(all_results).sort_values('ROI %', ascending=False)
            st.success("High-Precision Scan Complete!")
            
            # Show the "Golden Peak"
            st.subheader("🏆 The Global Top 10 High-Precision Strategies")
            st.dataframe(res_df.head(10), use_container_width=True)

            # Export
            csv_data = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full 3D Precision Results", data=csv_data, file_name=f"{ticker}_precision_3D.csv")

    # --- MODULE 2: TRADE DETAILER ---
    elif mode == "Trade Detailer":
        st.header("📜 High-Precision Trade Ledger")
        c1, c2, c3, c4 = st.columns(4)
        in_rsi = c1.number_input("RSI Length", value=14)
        in_ent = c2.number_input("Entry RSI", value=60)
        in_ext = c3.number_input("Exit RSI", value=50)
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
                for _, row in t_df.iterrows(): daily_rets.loc[pd.to_datetime(row['Exit Date'])] = row['P&L %'] / 100
                cum_strategy = (1 + daily_rets).cumprod()
                
                st.subheader("📊 Quant Scorecard")
                st.line_chart(100000 * cum_strategy)
                st.dataframe(t_df, use_container_width=True)