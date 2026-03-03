import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v6.2", layout="wide")

st.title("🛡️ Pro-Tracer: 4D Quant Optimizer (GMMA + RSI + RRR)")
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
    # --- MODULE 1: 4D OPTIMIZER (Now with RRR & Profit Factor) ---
    if mode == "Global 4D Optimizer":
        st.header("🔬 4D Strategic Quant Scan")
        if st.button("🚀 Run 4D Performance Scan"):
            all_results = []
            df = df_raw.copy()
            df['Market_Ret'] = df['Close'].pct_change()
            df['EMA_200'] = ta.ema(df['Close'], length=200)
            
            # Constraints
            len_range = range(3, 103, 20)  # RSI Lengths (Tightened for speed)
            ent_range = range(50, 61, 5)   # Entry 
            ext_range = range(40, 51, 5)   # Exit
            tsl_range = [5, 10, 15]        # Trailing SL %
            
            valid_combos = [(ent, ext, tsl) for ent in ent_range for ext in ext_range for tsl in tsl_range if ext < ent]
            total_combos = len(len_range) * len(valid_combos)
            progress_bar = st.progress(0)
            count = 0
            
            for r_len in len_range:
                rsi = ta.rsi(df['Close'], length=r_len)
                if rsi is None: continue
                for ent, ext, tsl in valid_combos:
                    count += 1
                    progress_bar.progress(min(count / total_combos, 1.0))
                    
                    in_pos, trade_pnls, entry_p, highest_p = False, [], 0, 0
                    
                    for j in range(1, len(df)):
                        curr_rsi, prev_rsi, curr_c = rsi.iloc[j], rsi.iloc[j-1], float(df['Close'].iloc[j])
                        if not in_pos:
                            if curr_rsi > ent and prev_rsi <= ent and curr_c > df['EMA_200'].iloc[j]:
                                in_pos, entry_p, highest_p = True, curr_c, curr_c
                        elif in_pos:
                            highest_p = max(highest_p, curr_c)
                            if curr_c <= (highest_p * (1 - tsl / 100)) or curr_rsi < ext:
                                in_pos = False
                                trade_pnls.append((curr_c - entry_p) / entry_p)
                    
                    if trade_pnls:
                        roi = (np.prod([1 + r for r in trade_pnls]) - 1) * 100
                        wins = [r for r in trade_pnls if r > 0]
                        losses = [r for r in trade_pnls if r <= 0]
                        win_rate = (len(wins) / len(trade_pnls)) * 100
                        avg_win = np.mean(wins) if wins else 0
                        avg_loss = abs(np.mean(losses)) if losses else 0
                        rrr = avg_win / avg_loss if avg_loss != 0 else np.inf
                        profit_factor = sum(wins) / abs(sum(losses)) if losses else np.inf
                        
                        all_results.append({
                            'RSI_Len': r_len, 'Entry': ent, 'Exit': ext, 'TSL %': tsl,
                            'ROI %': round(roi, 2), 'Win Rate %': round(win_rate, 1),
                            'Avg RRR': round(rrr, 2), 'Profit Factor': round(profit_factor, 2),
                            'Trades': len(trade_pnls)
                        })
            
            st.dataframe(pd.DataFrame(all_results).sort_values('ROI %', ascending=False), use_container_width=True)

    # --- MODULE 2: TRADE DETAILER (RE-INTEGRATED RRR & PROFIT FACTOR) ---
    elif mode == "Trade Detailer":
        st.header("📜 Full Quant Audit")
        c1, c2, c3, c4, c5 = st.columns(5)
        in_rsi = c1.number_input("RSI Length", value=13)
        in_ent = c2.slider("RSI Entry", 50, 65, 55)
        in_ext = c3.slider("RSI Exit", 35, 55, 45)
        tsl_val = c4.number_input("Trailing SL %", value=10.0)
        rvol_val = c5.number_input("Min RVOL (x)", value=1.2)
        
        if st.button("📊 Generate Complete Report"):
            df = df_raw.copy()
            df['RSI'] = ta.rsi(df['Close'], length=in_rsi)
            df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
            df['RVOL'] = df['Volume'] / df['Vol_MA']
            df['ST_Avg'] = df['Close'].rolling(window=8).mean() # Guppy ST Group Proxy
            df['LT_Avg'] = df['Close'].rolling(window=45).mean() # Guppy LT Group Proxy
            df['EMA_200'] = ta.ema(df['Close'], length=200)

            trades, in_trade, highest_p = [], False, 0
            for i in range(1, len(df)):
                curr_p, rsi_v, prev_rsi = float(df['Close'].iloc[i]), df['RSI'].iloc[i], df['RSI'].iloc[i-1]
                if not in_trade:
                    if rsi_v > in_ent and prev_rsi <= in_ent and df['ST_Avg'].iloc[i] > df['LT_Avg'].iloc[i] and df['RVOL'].iloc[i] >= rvol_val and curr_p > df['EMA_200'].iloc[i]:
                        in_trade, entry_date, entry_price = True, df.index[i], curr_p
                        highest_p = curr_p
                elif in_trade:
                    highest_p = max(highest_p, curr_p)
                    if curr_p <= (highest_p * (1 - tsl_val / 100)) or rsi_v < in_ext:
                        in_trade = False
                        pnl = ((curr_p - entry_price) / entry_price) * 100
                        trades.append({"Entry": entry_date.date(), "Exit": df.index[i].date(), "Entry P": round(entry_price, 2), "Exit P": round(curr_p, 2), "P&L %": round(pnl, 2), "Reason": "TSL" if curr_p <= (highest_p * (1 - tsl_val / 100)) else "RSI"})

            if trades:
                t_df = pd.DataFrame(trades)
                wins = t_df[t_df['P&L %'] > 0]['P&L %']
                losses = t_df[t_df['P&L %'] <= 0]['P&L %']
                rrr = wins.mean() / abs(losses.mean()) if not losses.empty else np.inf
                pf = wins.sum() / abs(losses.sum()) if not losses.empty else np.inf
                
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric("Win Rate", f"{(len(wins)/len(t_df))*100:.1f}%")
                s2.metric("Avg RRR", f"{rrr:.2f}:1")
                s3.metric("Profit Factor", f"{pf:.2f}")
                s4.metric("Trades", len(t_df))
                s5.metric("Avg P&L %", f"{t_df['P&L %'].mean():.2f}%")
                
                st.line_chart((1 + t_df['P&L %']/100).cumprod())
                st.dataframe(t_df, use_container_width=True)