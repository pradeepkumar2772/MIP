import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import datetime

# --- Streamlit UI Setup ---
st.title("🛡️ Pro-Tracer: Ultra-Optimizer")
st.markdown("### RSI vs EMA Brute-Force Engine (Range 3 - 252)")

# 1. User Inputs
ticker = st.text_input("Ticker Symbol", "TRENT.NS")

col1, col2 = st.columns(2)
start_date = col1.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = col2.date_input("End Date", datetime.date(2026, 3, 1))

# 2. Optimization Parameters
rsi_min, rsi_max = 3, 252
ema_min, ema_max = 3, 50 # EMA above 50 on an RSI is rarely effective, capped for speed

if st.button("🚀 Start Full Optimization"):
    # Download Data
    df = yf.download(ticker, start=start_date, end=end_date)
    df.columns = df.columns.get_level_values(0)
    
    if df.empty:
        st.error("No data found for this range.")
    else:
        results = []
        progress_bar = st.progress(0)
        
        # Calculate market returns once
        df['Market_Ret'] = df['Close'].pct_change()
        
        # 3. The Brute Force Loop
        total_steps = (rsi_max - rsi_min + 1)
        for i, r_len in enumerate(range(rsi_min, rsi_max + 1)):
            # Update progress every 10 steps
            if i % 10 == 0:
                progress_bar.progress(i / total_steps)
                
            # Pre-calculate RSI for this length
            rsi_series = ta.rsi(df['Close'], length=r_len)
            
            for e_len in range(ema_min, ema_max + 1):
                # Calculate EMA of RSI
                rsi_ema = ta.ema(rsi_series, length=e_len)
                
                # Strategy: Entry when RSI > EMA | Exit when RSI < EMA
                # shift(1) avoids look-ahead bias
                signal = (rsi_series > rsi_ema).astype(int)
                strat_ret = df['Market_Ret'] * signal.shift(1)
                
                # Performance Metric
                total_return = (1 + strat_ret).prod() - 1
                
                results.append({
                    'RSI_Len': r_len,
                    'EMA_Len': e_len,
                    'ROI_%': total_return * 100
                })
        
        progress_bar.progress(1.0)
        
        # 4. Display Results
        res_df = pd.DataFrame(results)
        best = res_df.loc[res_df['ROI_%'].idxmax()]
        
        st.success(f"Best Found: RSI {int(best['RSI_Len'])} | EMA {int(best['EMA_Len'])} | ROI: {best['ROI_%']:.2f}%")
        
        # Show Top 10
        st.write("### Top 10 Parameter Combinations")
        st.dataframe(res_df.sort_values('ROI_%', ascending=False).head(10))
        
        # Heatmap Visualization
        st.write("### Profitability Distribution")
        pivot = res_df.pivot(index="RSI_Len", columns="EMA_Len", values="ROI_%")
        st.line_chart(pivot) # Quick visualization of ROI across RSI lengths