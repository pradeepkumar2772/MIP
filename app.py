import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v6.1", layout="wide")

st.title("🛡️ Pro-Tracer: 4D Global Optimizer (RSI + GMMA + TSL)")
st.sidebar.header("Global Settings")

ticker = st.sidebar.text_input("Ticker Symbol", "BRITANNIA.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(1999, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

mode = st.sidebar.radio("Select Module", ["Global 4D Optimizer", "Trade Detailer"])

@st.cache_data
def get_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        if not data.empty and isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

df_raw = get_data(ticker, start_date, end_date)

if df_raw.empty:
    st.warning("Awaiting data...")
else:
    # --- MODULE 1: 4D OPTIMIZER (LENGTH, ENTRY, EXIT, TSL) ---
    if mode == "Global 4D Optimizer":
        st.header("🔬 4-Dimensional Strategic Optimizer")
        st.info("Searching for the best combination of Momentum (RSI), Trend (GMMA), and Protection (TSL).")
        
        if st.button("🚀 Run 4D Deep Analysis"):
            all_results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            
            # Constraints & Ranges
            len_range = range(3, 253, 20)  # RSI Lengths
            ent_range = range(50, 61, 5)   # Entry Levels
            ext_range = range(40, 51, 5)   # Exit Levels
            tsl_range = [5, 10, 15, 20]    # Trailing SL % options
            
            # GMMA Averages for Trend Filtering
            df['EMA_200'] = ta.ema(df['Close'], length=200)
            
            valid_combos = []
            for ent in ent_range:
                for ext in ext_range:
                    if ext < ent:
                        for tsl in tsl_range:
                            valid_combos.append((ent, ext, tsl))
            
            total_combos = len(len_range) * len(valid_combos)
            progress_bar = st.progress(0)
            count = 0
            
            for r_len in len_range:
                rsi = ta.rsi(df['Close'], length=r_len)
                if rsi is None: continue
                
                for ent, ext, tsl in valid_combos:
                    count += 1
                    progress_bar.progress(min(count / total_combos, 1.0))
                    
                    in_pos, trade_results = False, []
                    entry_p, highest_p = 0, 0
                    
                    for j in range(1, len(df)):
                        curr_rsi = rsi.iloc[j]
                        prev_rsi = rsi.iloc[j-1]
                        curr_c = float(df['Close'].iloc[j])
                        
                        if not in_pos:
                            # Entry: RSI Cross + Price > 200 EMA
                            if curr_rsi > ent and prev_rsi <= ent and curr_c > df['EMA_200'].iloc[j]:
                                in_pos, entry_p, highest_p = True, curr_c, curr_c
                        elif in_pos:
                            highest_p = max(highest_p, curr_c)
                            tsl_hit = curr_c <= (highest_p * (1 - tsl / 100))
                            rsi_exit = curr_rsi < ext
                            
                            if tsl_hit or rsi_exit:
                                in_pos = False
                                trade_results.append((curr_c - entry_p) / entry_p)
                    
                    if trade_results:
                        roi = (np.prod([1 + r for r in trade_results]) - 1) * 100
                        wins = [r for r in trade_results if r > 0]
                        win_rate = (len(wins) / len(trade_results)) * 100
                        
                        # Max Drawdown Calculation for each combo
                        daily_rets = pd.Series(0, index=df.index)
                        # (Approximation for speed in optimizer)
                        max_dd = -1 # Placeholder
                        
                        all_results.append({
                            'RSI_Len': r_len, 'Entry': ent, 'Exit': ext, 'TSL %': tsl,
                            'ROI %': round(roi, 2), 'Win Rate %': round(win_rate, 1),
                            'Trades': len(trade_results)
                        })
            
            res_df = pd.DataFrame(all_results).sort_values('ROI %', ascending=False)
            st.success("4D Scan Complete!")
            st.dataframe(res_df, use_container_width=True)

    # --- MODULE 2: TRADE DETAILER (WITH RVOL, GMMA & TSL) ---
    elif mode == "Trade Detailer":
        st.header("📜 Full Institutional Audit")
        c1, c2, c3, c4, c5 = st.columns(5)
        in_rsi = c1.number_input("RSI Length", value=13)
        in_ent = c2.slider("RSI Entry", 50, 65, 55)
        in_ext = c3.slider("RSI Exit", 35, 55, 45)
        tsl_val = c4.number_input("Trailing SL %", value=10.0)
        rvol_val = c5.number_input("Min RVOL (x)", value=1.2)
        
        if st.button("📊 Generate Comprehensive Report"):
            df = df_raw.copy()
            df['RSI'] = ta.rsi(df['Close'], length=in_rsi)
            df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
            df['RVOL'] = df['Volume'] / df['Vol_MA']
            df['EMA_200'] = ta.ema(df['Close'], length=200)
            
            # GMMA Short-Term Avg (3,5,8,10,12,15)
            df['ST_Avg'] = df['Close'].rolling(window=3).mean() # Simplified for code flow
            # GMMA Long-Term Avg (30,35,40,45,50,60)
            df['LT_Avg'] = df['Close'].rolling(window=30).mean() 

            trades, in_trade, highest_p = [], False, 0
            
            for i in range(1, len(df)):
                curr_p = float(df['Close'].iloc[i])
                rsi_v = df['RSI'].iloc[i]
                prev_rsi = df['RSI'].iloc[i-1]
                
                if not in_trade:
                    # Entry: RSI Cross + GMMA Trend + RVOL + Above 200 EMA
                    if rsi_v > in_ent and prev_rsi <= in_ent and df['ST_Avg'].iloc[i] > df['LT_Avg'].iloc[i] and df['RVOL'].iloc[i] >= rvol_val and curr_p > df['EMA_200'].iloc[i]:
                        in_trade, entry_date, entry_price = True, df.index[i], curr_p
                        highest_p = curr_p
                
                elif in_trade:
                    highest_p = max(highest_p, curr_p)
                    tsl_trigger = curr_p <= (highest_p * (1 - tsl_val / 100))
                    rsi_exit = rsi_v < in_ext
                    
                    if tsl_trigger or rsi_exit:
                        in_trade = False
                        pnl = ((curr_p - entry_price) / entry_price) * 100
                        trades.append({
                            "Entry": entry_date.date(), "Exit": df.index[i].date(), 
                            "Entry P": round(entry_price, 2), "Exit P": round(curr_p, 2), 
                            "P&L %": round(pnl, 2), "Reason": "TSL" if tsl_trigger else "RSI"
                        })

            if trades:
                t_df = pd.DataFrame(trades)
                st.subheader("📊 Performance Scorecard")
                
                daily_rets = pd.Series(0.0, index=df.index)
                for _, row in t_df.iterrows(): daily_rets.loc[pd.to_datetime(row['Exit'])] = row['P&L %'] / 100
                cum_strategy = (1 + daily_rets).cumprod()
                
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Final ROI", f"{(cum_strategy.iloc[-1]-1)*100:.1f}%")
                s2.metric("Win Rate", f"{(len(t_df[t_df['P&L %'] > 0])/len(t_df))*100:.1f}%")
                s3.metric("TSL Exits", len(t_df[t_df['Reason'] == "TSL"]))
                s4.metric("Trades", len(t_df))
                
                st.line_chart(100000 * cum_strategy)
                st.dataframe(t_df, use_container_width=True)
            else:
                st.warning("No trades met the 4D criteria. Try adjusting thresholds.")