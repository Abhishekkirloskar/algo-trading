"""
Performance monitoring for a strategy.

This builds the data the dashboard and the agent need:
  * a portfolio equity curve (simulated, vs. buy & hold)
  * rolling risk metrics (how it's doing *recently*, not just overall)
  * a trade log (every round-trip with its return and holding period)
  * health checks (is it underperforming / in a deep drawdown right now?)

"Simulated tracking": we run the strategy forward over historical daily data
and treat the result as the portfolio's track record. No broker keys needed.
"""

import numpy as np
import pandas as pd

from data import fetch_data
from strategies import STRATEGIES
from backtest import run_backtest, metrics, trade_stats

TRADING_DAYS = 252
ROLL_WINDOW = 63          # ~3 trading months for "recent" metrics
RECENT_WINDOW = 126       # ~6 trading months for underperformance checks


def rolling_sharpe(returns: pd.Series, window: int = ROLL_WINDOW) -> pd.Series:
    mean = returns.rolling(window).mean() * TRADING_DAYS
    vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return mean / vol


def build_report(ticker: str, strategy: str, start: str, end: str,
                 fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """Run the strategy and attach monitoring columns (drawdown, rolling Sharpe)."""
    base = fetch_data(ticker, start, end)
    fn = STRATEGIES[strategy]
    df = run_backtest(fn(base, fast=fast, slow=slow))

    df["drawdown"] = df["strat_equity"] / df["strat_equity"].cummax() - 1
    df["bh_drawdown"] = df["bh_equity"] / df["bh_equity"].cummax() - 1
    df["roll_sharpe"] = rolling_sharpe(df["strategy_return"])
    return df


def trade_log(df: pd.DataFrame, cost_per_trade: float = 0.0005) -> pd.DataFrame:
    """Return a DataFrame of round-trip trades: entry/exit dates, return, days held."""
    rows = []
    entry_price = entry_date = entry_i = None
    closes = df["Close"].values
    trades = df["trade"].values
    positions = df["position"].values
    idx = df.index
    for i in range(len(df)):
        if trades[i] == 1 and positions[i] == 1:
            entry_price, entry_date, entry_i = closes[i], idx[i], i
        elif trades[i] == 1 and positions[i] == 0 and entry_price is not None:
            ret = closes[i] / entry_price - 1 - 2 * cost_per_trade
            rows.append({
                "entry": entry_date.date(),
                "exit": idx[i].date(),
                "days": i - entry_i,
                "return": ret,
                "result": "win" if ret > 0 else "loss",
            })
            entry_price = None
    return pd.DataFrame(rows)


def current_status(df: pd.DataFrame) -> dict:
    """A snapshot of where things stand on the most recent bar."""
    last = df.iloc[-1]
    return {
        "asof": df.index[-1].date().isoformat(),
        "in_market": bool(last["position"] == 1),
        "price": float(last["Close"]),
        "drawdown": float(last["drawdown"]),
        "roll_sharpe": float(last["roll_sharpe"]) if pd.notna(last["roll_sharpe"]) else float("nan"),
    }


def health_check(df: pd.DataFrame,
                 dd_limit: float = -0.20,
                 underperf_margin: float = 0.05,
                 sharpe_floor: float = 0.0) -> list:
    """
    Look for signs the strategy is struggling *right now*. Returns a list of
    (severity, message) flags. Empty list = looks healthy.

    These are deliberately simple, transparent rules — not a black box. All
    three thresholds are caller-configurable (dashboard sliders / CLI flags).
    """
    flags = []

    cur_dd = df["drawdown"].iloc[-1]
    if cur_dd <= dd_limit:
        flags.append(("high", f"Current drawdown {cur_dd:.0%} is past the {dd_limit:.0%} limit."))

    # Recent (≈6 month) performance vs. buy & hold.
    recent = df.iloc[-RECENT_WINDOW:]
    if len(recent) > 5:
        s = recent["strat_equity"].iloc[-1] / recent["strat_equity"].iloc[0] - 1
        b = recent["bh_equity"].iloc[-1] / recent["bh_equity"].iloc[0] - 1
        if s < b - underperf_margin:
            flags.append(("medium",
                          f"Lagging buy & hold over ~6 months ({s:+.1%} vs {b:+.1%})."))

    rs = df["roll_sharpe"].iloc[-1]
    if pd.notna(rs) and rs < sharpe_floor:
        flags.append(("medium",
                      f"3-month rolling Sharpe ({rs:.2f}) is below the {sharpe_floor:.2f} floor."))

    return flags


def monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    """A year×month table of the strategy's monthly returns — for a heatmap."""
    import calendar

    r = df["strategy_return"]
    grouped = (1 + r).groupby([df.index.year, df.index.month]).prod() - 1
    table = grouped.unstack()
    table.columns = [calendar.month_abbr[m] for m in table.columns]
    table.index.name = "Year"
    return table


def summary_metrics(df: pd.DataFrame) -> dict:
    """Headline strategy + buy & hold metrics + trade stats, all formatted."""
    strat = metrics(df["strategy_return"], df["strat_equity"])
    bh = metrics(df["market_return"], df["bh_equity"])
    ts = trade_stats(df)
    return {"strategy": strat, "buy_hold": bh, "trades": ts}
