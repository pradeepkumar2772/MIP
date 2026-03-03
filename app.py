import pandas as pd
import numpy as np
import yfinance as yf

# ===============================
# CONFIGURATION
# ===============================
symbol = "AAPL"
start_date = "2010-01-01"
end_date = "2024-01-01"
transaction_cost = 0.0005   # 0.05%
slippage = 0.0005           # 0.05%

# ===============================
# DATA DOWNLOAD
# ===============================
df = yf.download(symbol, start=start_date, end=end_date)
df = df[['Close']].dropna()

# ===============================
# RSI FUNCTION (Institutional Safe)
# ===============================
def compute_rsi(price, period):
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ===============================
# PERFORMANCE METRICS
# ===============================
def performance_metrics(returns):
    returns = returns.dropna()
    equity = (1 + returns).cumprod()

    total_return = equity.iloc[-1] - 1
    years = len(returns) / 252
    cagr = (equity.iloc[-1]) ** (1 / years) - 1

    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()

    sharpe = np.sqrt(252) * returns.mean() / returns.std()
    sortino = np.sqrt(252) * returns.mean() / returns[returns < 0].std()

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Max Drawdown": max_dd,
        "Sharpe": sharpe,
        "Sortino": sortino
    }

# ===============================
# BACKTEST FUNCTION
# ===============================
def backtest(data, rsi_period):

    df = data.copy()

    df['RSI'] = compute_rsi(df['Close'], rsi_period)
    df['RSI_MA_9'] = df['RSI'].rolling(9).mean()

    # Entry
    entry = (
        (df['RSI'] > df['RSI_MA_9']) &
        (df['RSI'].shift(1) <= df['RSI_MA_9'].shift(1))
    )

    # Exit
    exit = (
        (df['RSI'] < df['RSI_MA_9']) &
        (df['RSI'].shift(1) >= df['RSI_MA_9'].shift(1))
    )

    df['signal'] = 0
    df.loc[entry, 'signal'] = 1
    df.loc[exit, 'signal'] = 0

    df['position'] = df['signal'].replace(to_replace=0, method='ffill')
    df['position'] = df['position'].fillna(0)

    df['returns'] = df['Close'].pct_change()

    # Apply slippage & cost on trade days
    trades = df['position'].diff().abs()
    cost = trades * (transaction_cost + slippage)

    df['strategy_returns'] = (
        df['position'].shift(1) * df['returns']
        - cost
    )

    metrics = performance_metrics(df['strategy_returns'])

    return metrics

# ===============================
# TRAIN / TEST SPLIT
# ===============================
split = int(len(df) * 0.7)
train = df.iloc[:split]
test = df.iloc[split:]

results = []

for period in range(3, 253):
    train_metrics = backtest(train, period)
    test_metrics = backtest(test, period)

    results.append([
        period,
        train_metrics["Sharpe"],
        test_metrics["Sharpe"],
        test_metrics["CAGR"],
        test_metrics["Max Drawdown"]
    ])

results_df = pd.DataFrame(
    results,
    columns=["RSI_Period", "Train Sharpe", "Test Sharpe", "Test CAGR", "Test MaxDD"]
)

results_df = results_df.sort_values("Test Sharpe", ascending=False)

print("\nTop 10 Stable RSI Periods (Out-of-Sample):\n")
print(results_df.head(10))