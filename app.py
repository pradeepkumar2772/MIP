import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# -----------------------------
# 1. Download Data
# -----------------------------
symbol = "AAPL"  # Change to your stock
start_date = "2015-01-01"
end_date = "2024-01-01"

df = yf.download(symbol, start=start_date, end=end_date)
df = df[['Close']].copy()

# -----------------------------
# 2. RSI Function
# -----------------------------
def calculate_rsi(data, period):
    delta = data.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain = pd.Series(gain).rolling(period).mean()
    loss = pd.Series(loss).rolling(period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# -----------------------------
# 3. Backtest Function
# -----------------------------
def backtest_rsi_strategy(df, rsi_period):
    
    data = df.copy()
    
    # Calculate RSI
    data['RSI'] = calculate_rsi(data['Close'], rsi_period)
    
    # RSI Moving Averages
    data['RSI_MA_9'] = data['RSI'].rolling(9).mean()
    data['RSI_MA_21'] = data['RSI'].rolling(21).mean()
    
    # Entry Condition
    entry = (
        (data['RSI'] > data['RSI_MA_9']) &
        (data['RSI'].shift(1) <= data['RSI_MA_9'].shift(1)) &
        (data['RSI_MA_9'] > data['RSI_MA_21'])
    )
    
    # Exit Condition
    exit = (
        (data['RSI'] < data['RSI_MA_9']) &
        (data['RSI'].shift(1) >= data['RSI_MA_9'].shift(1)) &
        (data['RSI_MA_9'] < data['RSI_MA_21'])
    )
    
    data['Position'] = 0
    data.loc[entry, 'Position'] = 1
    data.loc[exit, 'Position'] = 0
    
    data['Position'] = data['Position'].replace(to_replace=0, method='ffill')
    data['Position'] = data['Position'].fillna(0)
    
    # Returns
    data['Market_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Market_Return'] * data['Position'].shift(1)
    
    # Performance
    cumulative_return = (1 + data['Strategy_Return']).cumprod().iloc[-1]
    sharpe = (data['Strategy_Return'].mean() / data['Strategy_Return'].std()) * np.sqrt(252)
    
    return cumulative_return, sharpe, data

# -----------------------------
# 4. Optimize RSI Period 3–252
# -----------------------------
results = []

for period in range(3, 253):
    try:
        cum_ret, sharpe, _ = backtest_rsi_strategy(df, period)
        results.append([period, cum_ret, sharpe])
    except:
        continue

results_df = pd.DataFrame(results, columns=['RSI_Period', 'Cumulative_Return', 'Sharpe'])

# Sort by Sharpe Ratio
results_df = results_df.sort_values(by='Sharpe', ascending=False)

print("\nTop 10 RSI Periods by Sharpe Ratio:\n")
print(results_df.head(10))

# -----------------------------
# 5. Plot Best Strategy
# -----------------------------
best_period = results_df.iloc[0]['RSI_Period']
_, _, best_data = backtest_rsi_strategy(df, int(best_period))

plt.figure(figsize=(12,6))
plt.plot((1 + best_data['Strategy_Return']).cumprod(), label="Strategy")
plt.plot((1 + best_data['Market_Return']).cumprod(), label="Buy & Hold")
plt.legend()
plt.title(f"Best RSI Period: {best_period}")
plt.show()