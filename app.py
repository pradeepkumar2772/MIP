import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro-Tracer v4.1", layout="wide")

st.title("🛡️ Pro-Tracer: Total Strategy Optimizer (3D)")
st.sidebar.header("Scan Parameters")

ticker = st.sidebar.text_input("Ticker Symbol", "TRENT.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(1999, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

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
    st.header("🔬 3-Way Global Optimization")
    st.info("Searching Lengths (3-252), Entries (55-80), and Exits (35-60) simultaneously.")

    if st.button("🚀 Start Global Deep-Scan"):
        all_results = []
        df = df_raw.copy()
        df['Market_Ret'] = df['Close'].pct_change()
        
        # --- 3D Optimization Space ---
        # We use slightly larger steps for the Global Scan to maintain speed
        len_range = range(3, 253, 15)  # 17 lengths
        ent_range = range(55, 81, 5)   # 6 entries
        ext_range = range(35, 61, 5)   # 6 exits
        
        total_combos = len(len_range) * len(ent_range) * len(ext_range)
        progress_bar = st.progress(0)
        status = st.empty()
        count = 0

        for r_len in len_range:
            rsi = ta.rsi(df['Close'], length=r_len)
            for ent in ent_range:
                for ext in ext_range:
                    if ext >= ent: continue # Logical check
                    
                    count += 1
                    progress_bar.progress(count / total_combos)
                    status.text(f"Testing: Len {r_len} | Entry {ent} | Exit {ext}")

                    # Simulation
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
        st.success("Global Deep-Scan Complete!")
        
        # Display the "Holy Grail" top 5
        st.subheader("🏆 The Global Top 5 Strategies")
        st.table(res_df.head(5))

        # Show the rest in a filterable table
        st.write("### 📋 Full Strategy Database")
        st.dataframe(res_df, use_container_width=True)

        # Download button for the full scan results
        csv_data = res_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download All 3D Results", data=csv_data, file_name=f"{ticker}_global_opt.csv")