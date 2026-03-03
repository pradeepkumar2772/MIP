import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(layout="wide")

st.title("📊 Institutional RSI Research")

# =============================
# USER INPUT
# =============================
symbol = st.text_input("Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))

transaction_cost = 0.0005
slippage = 0.0005

# =============================
# DOWNLOAD DATA
# =============================
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

df = load_data(symbol, start_date, end_date)

if df.empty:
    st.error("❌ Data download failed. Possibly internet blocked.")
    st.stop()

df = df[['Close']].dropna()

st.write("Data Loaded:", df.shape)

# =============================
# RSI FUNCTION
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

    sharpe = 0
    if returns.std() != 0:
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

    df["RSI"] = compute_rsi(df["Close"], rsi_period)
    df["RSI_MA_9"] = df["RSI"].rolling(9).mean()

    # Entry
    entry = (
        (df["RSI"] > df["RSI_MA_9"]) &
        (df["RSI"].shift(1) <= df["RSI_MA_9"].shift(1))
    )

    # Exit
    exit = (
        (df["RSI"] < df["RSI_MA_9"]) &
        (df["RSI"].shift(1) >= df["RSI_MA_9"].shift(1))
    )

    df["signal"] = np.nan
    df.loc[entry, "signal"] = 1
    df.loc[exit, "signal"] = 0

    df["position"] = df["signal"].ffill().fillna(0)

    df["returns"] = df["Close"].pct_change()

    trades = df["position"].diff().abs()
    cost = trades * (transaction_cost + slippage)

    df["strategy_returns"] = (
        df["position"].shift(1) * df["returns"]
        - cost
    )

    return performance_metrics(df["strategy_returns"])


# =============================
# OPTIMIZATION
# =============================
if st.button("Run Optimization"):

    split = int(len(df) * 0.7)
    train = df.iloc[:split]
    test = df.iloc[split:]

    results = []

    with st.spinner("Running RSI optimization..."):
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

    if len(results) == 0:
        st.error("No valid results generated.")
        st.stop()

    results_df = pd.DataFrame(
        results,
        columns=["RSI_Period", "Train_Sharpe", "Test_Sharpe", "Test_CAGR", "Test_MaxDD"]
    )

    results_df = results_df.sort_values("Test_Sharpe", ascending=False)

    st.subheader("Top 10 RSI Periods (Out-of-Sample)")
    st.dataframe(results_df.head(10))

    st.success("Optimization Complete ✅")