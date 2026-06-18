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


def sma_crossover_trend(df: pd.DataFrame, fast: int = 20, slow: int = 50,
                        trend: int = 200) -> pd.DataFrame:
    """
    SMA crossover WITH a long-term trend filter.

    Same as `sma_crossover`, but we only allow long trades when the price is
    also above its long-term (default 200-day) moving average — i.e. only buy
    when the bigger trend is up. This skips counter-trend whipsaws, which are
    where crossover strategies rack up small losses, so it usually improves the
    win rate and reduces drawdown (often at the cost of fewer trades / a bit
    less raw return).
    """
    out = df.copy()
    out["sma_fast"] = out["Close"].rolling(window=fast).mean()
    out["sma_slow"] = out["Close"].rolling(window=slow).mean()
    out["sma_trend"] = out["Close"].rolling(window=trend).mean()

    long_cond = (out["sma_fast"] > out["sma_slow"]) & (out["Close"] > out["sma_trend"])
    out["position"] = long_cond.astype(int)
    out["position"] = out["position"].shift(1).fillna(0)  # no look-ahead
    return out


def _rsi(series: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index — a 0-100 momentum gauge. Low = oversold."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def rsi_mean_reversion(df: pd.DataFrame, fast: int = 5, slow: int = 200,
                       rsi_period: int = 2, entry: int = 10) -> pd.DataFrame:
    """
    A mean-reversion strategy (Connors RSI-2 style) — built for a HIGH win rate.

    Idea: in a longer-term uptrend, brief sharp dips tend to bounce. So:
      * Only trade when price is above its long-term average (`slow`, 200-day) — uptrend.
      * BUY when the short RSI (period 2) drops below `entry` (very oversold, e.g. <10).
      * SELL when price recovers back above its short (`fast`, 5-day) average.

    These trades are frequent, small, and win often — the opposite profile of
    the trend-following crossover. (Note: `fast`/`slow` are reused as the short
    exit average and the long trend filter so it plugs into the same CLI.)
    """
    out = df.copy()
    out["sma_fast"] = out["Close"].rolling(window=fast).mean()    # exit average
    out["sma_slow"] = out["Close"].rolling(window=slow).mean()    # trend filter
    out["rsi"] = _rsi(out["Close"], rsi_period)

    uptrend = out["Close"] > out["sma_slow"]
    enter = uptrend & (out["rsi"] < entry)
    exit_ = out["Close"] > out["sma_fast"]

    # Build the position statefully: once we enter, hold until an exit signal.
    pos = []
    holding = 0
    for en, ex in zip(enter.fillna(False), exit_.fillna(False)):
        if holding == 0 and en:
            holding = 1
        elif holding == 1 and ex:
            holding = 0
        pos.append(holding)
    out["position"] = pd.Series(pos, index=out.index).shift(1).fillna(0)
    return out


def donchian_breakout(df: pd.DataFrame, fast: int = 10, slow: int = 20) -> pd.DataFrame:
    """
    Donchian channel breakout — a classic SWING / trend strategy.

    * BUY when today's close breaks ABOVE the highest high of the prior `slow`
      days (e.g. a new 20-day high → momentum is breaking out).
    * SELL when the close drops BELOW the lowest low of the prior `fast` days
      (e.g. a 10-day low → momentum is fading).

    Using the PRIOR window (`.shift(1)`) means "break a level that already
    existed", not one that includes today — avoiding look-ahead. Holding
    periods land in the days-to-weeks range, which is the swing sweet spot.
    """
    out = df.copy()
    upper = out["High"].rolling(window=slow).max().shift(1)   # prior N-day high
    lower = out["Low"].rolling(window=fast).min().shift(1)    # prior M-day low

    enter = out["Close"] > upper
    exit_ = out["Close"] < lower

    pos = []
    holding = 0
    for en, ex in zip(enter.fillna(False), exit_.fillna(False)):
        if holding == 0 and en:
            holding = 1
        elif holding == 1 and ex:
            holding = 0
        pos.append(holding)
    out["position"] = pd.Series(pos, index=out.index).shift(1).fillna(0)
    return out


# A registry so backtest.py can pick a strategy by name from the command line.
STRATEGIES = {
    "sma_crossover": sma_crossover,
    "sma_crossover_trend": sma_crossover_trend,
    "rsi_mean_reversion": rsi_mean_reversion,
    "donchian_breakout": donchian_breakout,
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
