import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt

# 1. Download Data
ticker = "RELIANCE.NS"  # Using an Indian heavyweight for context
data = yf.download(ticker, start="2018-01-01", end="2026-03-01")

# 2. Calculate RSI
data['RSI'] = ta.rsi(data['Close'], length=14)

# 3. Generate Signals
# Strategy: Buy when RSI > 50 (bullish momentum) and exit when < 40
data['Signal'] = 0
data.loc[data['RSI'] > 50, 'Signal'] = 1
data.loc[data['RSI'] < 40, 'Signal'] = 0

# 4. Calculate Returns
# 'Shift' the signal by 1 day because we enter at the NEXT day's open
data['Market_Return'] = data['Close'].pct_change()
data['Strategy_Return'] = data['Market_Return'] * data['Signal'].shift(1)

# 5. Calculate Metrics
data['Cumulative_Market'] = (1 + data['Market_Return']).cumprod()
data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()

# Calculate Max Drawdown
rolling_max = data['Cumulative_Strategy'].cummax()
drawdown = (data['Cumulative_Strategy'] - rolling_max) / rolling_max
max_drawdown = drawdown.min()

print(f"Total Strategy Return: {(data['Cumulative_Strategy'].iloc[-1] - 1) * 100:.2f}%")
print(f"Max Drawdown: {max_drawdown * 100:.2f}%")