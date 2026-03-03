import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st

def validate_strategy(ticker):
    df = yf.download(ticker, start="2020-01-01")
    df.columns = df.columns.get_level_values(0)
    
    # Split Data: 70% Training, 30% Testing
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    st.info(f"Training on: {train_df.index[0].date()} to {train_df.index[-1].date()}")
    st.info(f"Validating on: {test_df.index[0].date()} to {test_df.index[-1].date()}")

    # --- Phase 1: Optimization (In-Sample) ---
    best_ret = -999
    best_rsi = 14
    best_sig = 9

    for r_len in range(10, 101, 10):
        rsi_series = ta.rsi(train_df['Close'], length=r_len)
        for s_len in range(5, 31, 5):
            rsi_sma = ta.sma(rsi_series, length=s_len)
            signal = (rsi_series > rsi_sma).astype(int)
            
            returns = train_df['Close'].pct_change()
            strat_rets = (returns * signal.shift(1)).fillna(0)
            total_ret = (1 + strat_rets).prod() - 1
            
            if total_ret > best_ret:
                best_ret = total_ret
                best_rsi = r_len
                best_sig = s_len

    # --- Phase 2: Validation (Out-of-Sample) ---
    # Apply BEST parameters to the hidden TEST data
    test_rsi_series = ta.rsi(test_df['Close'], length=best_rsi)
    test_rsi_sma = ta.sma(test_rsi_series, length=best_sig)
    test_signal = (test_rsi_series > test_rsi_sma).astype(int)
    
    test_returns = test_df['Close'].pct_change()
    test_strat_rets = (test_returns * test_signal.shift(1)).fillna(0)
    validation_ret = (1 + test_strat_rets).prod() - 1

    return best_rsi, best_sig, best_ret * 100, validation_ret * 100

# --- Streamlit UI ---
st.title("Pro-Tracer: Walk-Forward Validator")
ticker = st.text_input("Ticker", "TRENT.NS")

if st.button("Run Rigorous Validation"):
    br, bs, tr_ret, val_ret = validate_strategy(ticker)
    
    col1, col2 = st.columns(2)
    col1.metric("Best Training Return", f"{tr_ret:.2f}%", help="Results from known data")
    col2.metric("Validation Return", f"{val_ret:.2f}%", help="Results from hidden data", 
                delta=f"{val_ret - tr_ret:.2f}%")
    
    st.success(f"Optimized Parameters: RSI {br} | Signal {bs}")
    
    if val_ret > 0:
        st.balloons()
        st.write("✅ **Strategy is Robust:** It survived the hidden data test.")
    else:
        st.error("❌ **Overfitting Detected:** Strategy failed on unseen data.")