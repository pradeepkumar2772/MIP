import pandas as pd
import numpy as np
import yfinance as yf

# =============================
# CONFIGURATION
# =============================
symbol = "AAPL"
start_date = "2015-01-01"
end_date = "2024-01-01"

transaction_cost = 0.0005   # 0.05%
slippage = 0.0005           # 0.05%

# =============================
# DOWNLOAD DATA
# =============================
df = yf.download(symbol, start=start_date, end=end_date)

# Fix possible MultiIndex columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[['Close']].dropna()

# =============================
# RSI FUNCTION (Wilder / RMA)
# =============================
def compute_rsi(price, period):
    delta = price.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


# =============================
# PERFORMANCE METRICS
# =============================
def performance_metrics(returns):

    returns = returns.dropna()

    if len(returns) < 50:
        return None

    equity = (1 + returns).cumprod()

    years = len(returns) / 252
    cagr = equity.iloc[-1]**(1/years) - 1 if years > 0 else 0

    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()

    if returns.std() == 0:
        sharpe = 0
    else:
        sharpe = np.sqrt(252) * returns.mean() / returns.std()

    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": max_dd
    }


# =============================
# BACKTEST FUNCTION
# =============================
def backtest(data, rsi_period):

    df = data.copy()

    # Indicators
    df["RSI"] = compute_rsi(df["Close"], rsi_period)
    df["RSI_MA_9"] = df["RSI"].rolling(9).mean()
    df["RSI_MA_21"] = df["RSI"].rolling(21).mean()

    # Entry: RSI crosses above MA9
    entry = (
        (df["RSI"] > df["RSI_MA_9"]) &
        (df["RSI"].shift(1) <= df["RSI_MA_9"].shift(1))
    )

    # Exit: RSI crosses below MA9
    exit = (
        (df["RSI"] < df["RSI_MA_9"]) &
        (df["RSI"].shift(1) >= df["RSI_MA_9"].shift(1))
    )

    # Signal column
    df["signal"] = np.nan
    df.loc[entry, "signal"] = 1
    df.loc[exit, "signal"] = 0

    # Forward fill positions (NO deprecated replace)
    df["position"] = df["signal"].ffill().fillna(0)

    # Market returns
    df["returns"] = df["Close"].pct_change()

    # Trade detection
    trades = df["position"].diff().abs()

    cost = trades * (transaction_cost + slippage)

    # Strategy returns
    df["strategy_returns"] = (
        df["position"].shift(1) * df["returns"]
        - cost
    )

    return performance_metrics(df["strategy_returns"])


# =============================
# TRAIN / TEST SPLIT
# =============================
split = int(len(df) * 0.7)

train = df.iloc[:split]
test = df.iloc[split:]

results = []

for period in range(3, 253):

    train_metrics = backtest(train, period)
    test_metrics = backtest(test, period)

    if train_metrics is None or test_metrics is None:
        continue

    results.append([
        period,
        train_metrics["Sharpe"],
        test_metrics["Sharpe"],
        test_metrics["CAGR"],
        test_metrics["MaxDD"]
    ])

results_df = pd.DataFrame(
    results,
    columns=["RSI_Period", "Train_Sharpe", "Test_Sharpe", "Test_CAGR", "Test_MaxDD"]
)

results_df = results_df.sort_values("Test_Sharpe", ascending=False)

print("\nTop 10 RSI Periods (Out-of-Sample Ranked):\n")
print(results_df.head(10))