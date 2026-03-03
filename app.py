import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st

def run_optimization(ticker):
    df = yf.download(ticker, start="2020-01-01")
    df.columns = df.columns.get_level_values(0)
    
    results = []

    # Optimization Loop: Testing RSI lengths from 3 to 252
    # To save time, we can step by 5 or 10, or run the full range
    for rsi_len in range(3, 253, 5): 
        temp_df = df.copy()
        temp_df['RSI'] = ta.rsi(temp_df['Close'], length=rsi_len)
        
        # Strategy: Buy > 50, Exit < 40
        temp_df['Signal'] = 0
        temp_df.loc[temp_df['RSI'] > 50, 'Signal'] = 1
        
        temp_df['Returns'] = temp_df['Close'].pct_change()
        temp_df['Strat_Ret'] = temp_df['Returns'] * temp_df['Signal'].shift(1)
        
        total_ret = (1 + temp_df['Strat_Ret']).prod() - 1
        win_rate = len(temp_df[temp_df['Strat_Ret'] > 0]) / len(temp_df[temp_df['Strat_Ret'] != 0]) if len(temp_df[temp_df['Strat_Ret'] != 0]) > 0 else 0
        
        results.append({
            'RSI_Length': rsi_len,
            'Total_Return': total_ret * 100,
            'Win_Rate': win_rate * 100
        })
    
    return pd.DataFrame(results)

# Streamlit UI
st.title("Pro-Tracer: RSI Parameter Optimizer")
ticker_input = st.text_input("Enter Ticker", "TRENT.NS")

if st.button("Start Brute-Force Optimization"):
    opt_results = run_optimization(ticker_input)
    
    # 2. Visualizing the "Profit Plateau"
    st.write("### Optimization Results (RSI Length vs Return)")
    st.line_chart(opt_results.set_index('RSI_Length')['Total_Return'])
    
    best_params = opt_results.loc[opt_results['Total_Return'].idxmax()]
    st.success(f"Best RSI Length: {best_params['RSI_Length']} with {best_params['Total_Return']:.2f}% Return")