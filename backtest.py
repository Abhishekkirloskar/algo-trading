"""
A simple, educational stock-strategy backtester.

WHAT IS BACKTESTING?
Running a trading strategy on HISTORICAL data to see how it *would* have
performed — without risking any real money. It's the first step every algo
trader takes: if a strategy didn't work on the past, there's no reason to
trust it with the future.

This script:
  1. Downloads historical prices (US or Indian stocks)
  2. Runs a strategy to decide when to be in/out of the market
  3. Compares the strategy to simply buying and holding the stock
  4. Prints performance metrics and saves a chart

USAGE
  python backtest.py                         # default: AAPL, SMA crossover
  python backtest.py --ticker RELIANCE.NS    # an Indian (NSE) stock
  python backtest.py --ticker TCS.NS --fast 10 --slow 30 --start 2018-01-01

Nothing here places real orders. It's a simulation.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import fetch_data
from strategies import STRATEGIES

# Trading days in a year — used to annualize returns/volatility.
TRADING_DAYS = 252


def run_backtest(df: pd.DataFrame, cost_per_trade: float = 0.0005) -> pd.DataFrame:
    """
    Turn a DataFrame that already has a 'position' column into daily returns.

    `cost_per_trade` models real-world friction (brokerage + slippage) as a
    fraction of the trade value, charged whenever the position changes.
    0.0005 = 0.05%, a rough stand-in for retail costs.
    """
    out = df.copy()

    # Daily return of just holding the stock (buy & hold benchmark).
    out["market_return"] = out["Close"].pct_change().fillna(0)

    # Our strategy only earns the market return on days we're holding (pos=1).
    out["strategy_return"] = out["position"] * out["market_return"]

    # Subtract trading costs on days the position flips (a buy or a sell).
    trades = out["position"].diff().abs().fillna(0)
    out["strategy_return"] -= trades * cost_per_trade
    out["trade"] = trades  # 1 on days we entered or exited

    # Equity curves: how $1 grows over time (compounded).
    out["bh_equity"] = (1 + out["market_return"]).cumprod()
    out["strat_equity"] = (1 + out["strategy_return"]).cumprod()
    return out


def trade_stats(df: pd.DataFrame, cost_per_trade: float = 0.0005) -> dict:
    """
    Break the run into round-trip trades (a BUY followed by its SELL) and
    measure how many were winners.

    Win rate = profitable round-trips / total round-trips. Also report the
    average win and average loss, which matter just as much: you can win only
    40% of the time and still be profitable if your wins are bigger.
    """
    trade_returns = []
    holds = []                       # how many trading days each trade lasted
    entry_price = None
    entry_i = None
    closes = df["Close"].values
    trades = df["trade"].values
    positions = df["position"].values
    for i in range(len(df)):
        if trades[i] == 1 and positions[i] == 1:            # entered (bought)
            entry_price = closes[i]
            entry_i = i
        elif trades[i] == 1 and positions[i] == 0 and entry_price is not None:  # exited
            gross = closes[i] / entry_price - 1
            trade_returns.append(gross - 2 * cost_per_trade)  # buy + sell costs
            holds.append(i - entry_i)
            entry_price = None

    if not trade_returns:
        return {"Round-trip trades": "0", "Win rate": "—", "Avg win": "—",
                "Avg loss": "—", "Avg hold (days)": "—"}

    tr = pd.Series(trade_returns)
    wins = tr[tr > 0]
    losses = tr[tr <= 0]
    return {
        "Round-trip trades": f"{len(tr)}",
        "Win rate": f"{len(wins) / len(tr):.0%}",
        "Avg win": f"{wins.mean():+.1%}" if len(wins) else "—",
        "Avg loss": f"{losses.mean():+.1%}" if len(losses) else "—",
        "Avg hold (days)": f"{sum(holds) / len(holds):.0f}",
    }


def metrics(returns: pd.Series, equity: pd.Series) -> dict:
    """Compute the headline performance numbers from a return series."""
    total_return = equity.iloc[-1] - 1
    years = len(returns) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan

    # Sharpe ratio: return per unit of risk (volatility). Higher is better;
    # >1 is decent, >2 is very good. Assumes a 0% risk-free rate for simplicity.
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (returns.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan

    # Max drawdown: the worst peak-to-trough drop. How much pain you'd endure.
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    return {
        "Total return": f"{total_return:7.1%}",
        "CAGR (per yr)": f"{cagr:7.1%}",
        "Sharpe ratio": f"{sharpe:7.2f}",
        "Max drawdown": f"{max_dd:7.1%}",
    }


def plot(df: pd.DataFrame, ticker: str, out_path: str):
    """Save a 2-panel chart: price + moving averages, and the equity curves."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(df.index, df["Close"], label="Close", color="black", linewidth=1)
    if "sma_fast" in df:
        ax1.plot(df.index, df["sma_fast"], label="Fast SMA", color="tab:blue", linewidth=1)
    if "sma_slow" in df:
        ax1.plot(df.index, df["sma_slow"], label="Slow SMA", color="tab:orange", linewidth=1)

    # Mark buys (entering) and sells (exiting) on the price chart.
    buys = df[(df["trade"] == 1) & (df["position"] == 1)]
    sells = df[(df["trade"] == 1) & (df["position"] == 0)]
    ax1.scatter(buys.index, buys["Close"], marker="^", color="green", s=60, label="Buy", zorder=5)
    ax1.scatter(sells.index, sells["Close"], marker="v", color="red", s=60, label="Sell", zorder=5)
    ax1.set_title(f"{ticker} — price & signals")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(df.index, df["bh_equity"], label="Buy & Hold", color="gray")
    ax2.plot(df.index, df["strat_equity"], label="Strategy", color="tab:green")
    ax2.set_title("Growth of $1 (strategy vs. buy & hold)")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nChart saved to: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Simple stock strategy backtester")
    p.add_argument("--ticker", default="AAPL", help="e.g. AAPL or RELIANCE.NS")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--strategy", default="sma_crossover", choices=list(STRATEGIES))
    p.add_argument("--fast", type=int, default=20, help="fast SMA window")
    p.add_argument("--slow", type=int, default=50, help="slow SMA window")
    args = p.parse_args()

    # 1. Data
    df = fetch_data(args.ticker, args.start, args.end)

    # 2. Strategy -> positions
    strat_fn = STRATEGIES[args.strategy]
    df = strat_fn(df, fast=args.fast, slow=args.slow)

    # 3. Simulate
    df = run_backtest(df)

    # 4. Report
    strat = metrics(df["strategy_return"], df["strat_equity"])
    bh = metrics(df["market_return"], df["bh_equity"])
    n_trades = int(df["trade"].sum())

    print("\n" + "=" * 52)
    print(f"  {args.ticker}   {args.start} -> {args.end}")
    print(f"  Strategy: {args.strategy} (fast={args.fast}, slow={args.slow})")
    print("=" * 52)
    print(f"  {'Metric':<16}{'Strategy':>12}{'Buy & Hold':>14}")
    print("  " + "-" * 42)
    for k in strat:
        print(f"  {k:<16}{strat[k]:>12}{bh[k]:>14}")
    print(f"  {'Trades made':<16}{n_trades:>12}{'—':>14}")
    print("=" * 52)

    # Per-trade win/loss breakdown for the strategy.
    ts = trade_stats(df)
    print("  Strategy trade breakdown:")
    for k, v in ts.items():
        print(f"  {k:<20}{v:>10}")
    print("=" * 52)
    print("  Note: past performance does NOT guarantee future results.")

    plot(df, args.ticker, out_path="backtest_result.png")


if __name__ == "__main__":
    main()
