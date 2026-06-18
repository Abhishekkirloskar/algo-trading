"""
Trading strategies.

A "strategy" here is just a function that looks at price data and decides,
for each day, whether we want to be IN the market (holding the stock) or OUT
(holding cash). We express that as a `position` column:

    position = 1  ->  hold the stock (we're "long")
    position = 0  ->  hold cash (we're "flat")

The backtester then applies those positions to the daily returns to see how
the strategy would have performed.

NEW TO TRADING? Two core ideas used below:

  * Moving average (MA): the average closing price over the last N days. It
    smooths out daily noise so you can see the trend. A "fast" MA (e.g. 20
    days) reacts quickly; a "slow" MA (e.g. 50 days) reacts slowly.

  * Crossover: when the fast MA crosses ABOVE the slow MA, recent prices are
    rising faster than the longer trend — a common signal to BUY. When it
    crosses BELOW, momentum is fading — a signal to SELL (go to cash).
"""

import pandas as pd


def sma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """
    Simple Moving Average crossover strategy.

    Be long when the fast SMA is above the slow SMA, otherwise hold cash.
    Returns a copy of `df` with 'sma_fast', 'sma_slow', and 'position' columns.
    """
    out = df.copy()
    out["sma_fast"] = out["Close"].rolling(window=fast).mean()
    out["sma_slow"] = out["Close"].rolling(window=slow).mean()

    # 1 when fast > slow (uptrend), else 0. NaNs (early days) become 0 = flat.
    out["position"] = (out["sma_fast"] > out["sma_slow"]).astype(int)

    # IMPORTANT: shift the position by one day. We can only act on a signal
    # the day AFTER we see it (you can't trade on today's close before it
    # exists). Skipping this is a classic backtesting mistake called
    # "look-ahead bias" that makes results look unrealistically good.
    out["position"] = out["position"].shift(1).fillna(0)
    return out


# A registry so backtest.py can pick a strategy by name from the command line.
STRATEGIES = {
    "sma_crossover": sma_crossover,
}


def latest_signal(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> dict:
    """
    Compute the CURRENT desired position from the most recent data — for live
    / paper trading (as opposed to backtesting the whole history).

    Unlike the backtest we do NOT shift the signal: in live trading the latest
    completed bar is real information we can act on right now.

    Returns a dict with the decision and the numbers behind it.
    """
    sma_fast = df["Close"].rolling(window=fast).mean().iloc[-1]
    sma_slow = df["Close"].rolling(window=slow).mean().iloc[-1]
    want_long = bool(sma_fast > sma_slow)
    return {
        "want_long": want_long,            # True = hold the stock, False = cash
        "close": float(df["Close"].iloc[-1]),
        "sma_fast": float(sma_fast),
        "sma_slow": float(sma_slow),
        "asof": df.index[-1].date().isoformat(),
    }
