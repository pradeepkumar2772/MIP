import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v5.4", layout="wide")

st.title("🛡️ Pro-Tracer: Institutional RVOL & Stable UI")
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
    # --- MODULE 1: OPTIMIZER (FIXED PROGRESS BAR) ---
    if mode == "Global 3D Optimizer":
        st.header("🔬 Strategic Risk-Reward Optimizer")
        if st.button("🚀 Run Deep Analysis"):
            all_results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            
            # Constraints
            len_range = range(3, 253, 10) 
            ent_range = range(50, 61, 2)  
            ext_range = range(40, 61, 2)  
            
            # DYNAMIC TOTAL CALCULATION (Fixes the Error)
            valid_combos = []
            for ent in ent_range:
                for ext in ext_range:
                    if ext < ent:
                        valid_combos.append((ent, ext))
            
            total_combos = len(len_range) * len(valid_combos)
            progress_bar = st.progress(0)
            count = 0
            
            for r_len in len_range:
                rsi = ta.rsi(df['Close'], length=r_len)
                if rsi is None: continue
                for ent, ext in valid_combos:
                    count += 1
                    progress_bar.progress(min(count / total_combos, 1.0)) # Clamp to 1.0
                    
                    in_pos, trade_results = False, []
                    entry_p = 0
                    for j in range(1, len(df)):
                        val, curr_c = rsi.iloc[j], float(df['Close'].iloc[j])
                        if not in_pos and val > ent: in_pos, entry_p = True, curr_c
                        elif in_pos and val < ext: in_pos, _ = False, trade_results.append((curr_c - entry_p) / entry_p)
                    
                    if trade_results:
                        roi = (np.prod([1 + r for r in trade_results]) - 1) * 100
                        wins = [r for r in trade_results if r > 0]
                        losses = [r for r in trade_results if r <= 0]
                        win_rate = (len(wins) / len(trade_results)) * 100
                        avg_win = np.mean(wins) if wins else 0
                        avg_loss = abs(np.mean(losses)) if losses else 0
                        rrr = avg_win / avg_loss if avg_loss != 0 else np.inf
                        all_results.append({'RSI_Len': r_len, 'Entry': ent, 'Exit': ext, 'ROI %': round(roi, 2), 'Win Rate %': round(win_rate, 1), 'Avg RRR': round(rrr, 2), 'Trades': len(trade_results)})
            
            st.success("Scan Complete!")
            st.dataframe(pd.DataFrame(all_results).sort_values('ROI %', ascending=False), use_container_width=True)

    # --- MODULE 2: TRADE DETAILER (WITH RVOL & TRAILING SL) ---
    elif mode == "Trade Detailer":
        st.header("📜 Institutional RVOL Audit")
        c1, c2, c3, c4, c5 = st.columns(5)
        in_rsi = c1.number_input("RSI Length", value=14)
        in_ent = c2.slider("Entry Level", 50, 60, 55)
        in_ext = c3.slider("Exit Level", 40, 60, 45)
        tsl_pct = c4.number_input("Trailing SL %", value=10.0)
        rvol_target = c5.number_input("Min RVOL (x)", value=1.5, help="Entry only if Volume > X times 20-day Average.")
        
        if st.button("📊 Generate Institutional Report"):
            df = df_raw.copy()
            df['RSI'] = ta.rsi(df['Close'], length=in_rsi)
            df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
            df['RVOL'] = df['Volume'] / df['Vol_MA']
            df['Trend_EMA'] = ta.ema(df['Close'], length=200)
            
            trades = []
            in_trade = False
            highest_p = 0
            
            for i in range(1, len(df)):
                curr_p = float(df['Close'].iloc[i])
                rsi_v = df['RSI'].iloc[i]
                prev_rsi = df['RSI'].iloc[i-1]
                rvol_v = df['RVOL'].iloc[i]
                
                if not in_trade:
                    # ENTRY: RSI Cross + Trend + RVOL (Institutional Check)
                    if rsi_v > in_ent and prev_rsi <= in_ent and curr_p > df['Trend_EMA'].iloc[i] and rvol_v >= rvol_target:
                        in_trade, entry_date, entry_price = True, df.index[i], curr_p
                        highest_p = curr_p
                
                elif in_trade:
                    highest_p = max(highest_p, curr_p)
                    tsl_trigger = curr_p <= (highest_p * (1 - tsl_pct / 100))
                    rsi_exit = rsi_v < in_ext and prev_rsi >= in_ext
                    
                    if tsl_trigger or rsi_exit:
                        in_trade = False
                        pnl = ((curr_p - entry_price) / entry_price) * 100
                        trades.append({
                            "Entry": entry_date.date(), "Exit": df.index[i].date(), 
                            "Entry P": round(entry_price, 2), "Exit P": round(curr_p, 2), 
                            "RVOL at Entry": round(df['RVOL'].loc[entry_date], 2),
                            "P&L %": round(pnl, 2), "Reason": "TSL" if tsl_trigger else "RSI"
                        })

            if trades:
                t_df = pd.DataFrame(trades)
                st.subheader("📊 Institutional Scorecard")
                
                daily_rets = pd.Series(0.0, index=df.index)
                for _, row in t_df.iterrows(): daily_rets.loc[pd.to_datetime(row['Exit'])] = row['P&L %'] / 100
                cum_strategy = (1 + daily_rets).cumprod()
                
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Final ROI", f"{(cum_strategy.iloc[-1]-1)*100:.1f}%")
                s2.metric("Avg RVOL", f"{t_df['RVOL at Entry'].mean():.2f}x")
                s3.metric("Win Rate", f"{(len(t_df[t_df['P&L %'] > 0])/len(t_df))*100:.1f}%")
                s4.metric("Trades", len(t_df))
                
                st.line_chart(100000 * cum_strategy)
                st.dataframe(t_df, use_container_width=True)
            else:
                st.warning("No trades met the RVOL criteria. Try lowering 'Min RVOL'.")