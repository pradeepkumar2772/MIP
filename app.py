import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v9.1", layout="wide")

st.title("🛡️ Pro-Tracer: Institutional GMMA & RSI Suite")
st.sidebar.header("Global Settings")

ticker = st.sidebar.text_input("Ticker Symbol", "BRITANNIA.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

mode = st.sidebar.radio("Select Module", ["Global 4D Optimizer", "Trade Detailer"])

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
    # --- MODULE 1: 4D OPTIMIZER ---
    if mode == "Global 4D Optimizer":
        st.header("🔬 4D Strategic Quant Scan")
        if st.button("🚀 Run 4D Performance Scan"):
            all_results = []
            df = df_raw.copy()
            df['EMA_200'] = ta.ema(df['Close'], length=200)
            
            # Constraints
            len_range = range(3, 103, 20) 
            ent_range = range(50, 61, 5)   
            ext_range = range(40, 51, 5)   
            tsl_range = [5, 10, 15]        
            
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
                        curr_rsi, curr_c = rsi.iloc[j], float(df['Close'].iloc[j])
                        if not in_pos:
                            if curr_rsi > ent and curr_c > df['EMA_200'].iloc[j]:
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
                        rrr = np.mean(wins) / abs(np.mean(losses)) if losses and wins else 0
                        all_results.append({
                            'RSI_Len': r_len, 'Entry': ent, 'Exit': ext, 'TSL %': tsl,
                            'ROI %': round(roi, 2), 'Win Rate %': round(len(wins)/len(trade_pnls)*100, 1),
                            'Avg RRR': round(rrr, 2), 'Trades': len(trade_pnls)
                        })
            st.dataframe(pd.DataFrame(all_results).sort_values('ROI %', ascending=False), use_container_width=True)

    # --- MODULE 2: TRADE DETAILER ---
    elif mode == "Trade Detailer":
        st.header("📜 Performance Audit & Ledger")
        c1, c2, c3, c4 = st.columns(4)
        in_rsi = c1.number_input("RSI Length", value=13)
        in_ent = c2.slider("RSI Entry", 50, 65, 55)
        in_ext = c3.slider("RSI Exit", 35, 55, 45)
        tsl_val = c4.number_input("Trailing SL %", value=10.0)
        
        if st.button("📊 Generate Report"):
            df = df_raw.copy()
            df['RSI'] = ta.rsi(df['Close'], length=in_rsi)
            df['ST_Avg'] = df['Close'].rolling(window=8).mean() 
            df['LT_Avg'] = df['Close'].rolling(window=45).mean()
            df['EMA_200'] = ta.ema(df['Close'], length=200)

            trades, in_trade, highest_p = [], False, 0
            for i in range(1, len(df)):
                curr_p, rsi_v = float(df['Close'].iloc[i]), df['RSI'].iloc[i]
                if not in_trade:
                    if rsi_v > in_ent and df['ST_Avg'].iloc[i] > df['LT_Avg'].iloc[i] and curr_p > df['EMA_200'].iloc[i]:
                        in_trade, entry_date, entry_price = True, df.index[i], curr_p
                        highest_p = curr_p
                elif in_trade:
                    highest_p = max(highest_p, curr_p)
                    if curr_p <= (highest_p * (1 - tsl_val / 100)) or rsi_v < in_ext:
                        in_trade = False
                        pnl = ((curr_p - entry_price) / entry_price) * 100
                        trades.append({"Entry": entry_date.date(), "Exit": df.index[i].date(), "P&L %": round(pnl, 2)})

            if trades:
                t_df = pd.DataFrame(trades)
                st.subheader("📊 Scorecard")
                s1, s2, s3 = st.columns(3)
                s1.metric("Total ROI", f"{t_df['P&L %'].sum():.1f}%")
                s2.metric("Win Rate", f"{(len(t_df[t_df['P&L %'] > 0])/len(t_df))*100:.1f}%")
                s3.metric("Trades", len(t_df))
                st.line_chart((1 + t_df['P&L %']/100).cumprod())
                st.dataframe(t_df, use_container_width=True)