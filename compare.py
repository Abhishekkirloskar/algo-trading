"""
Compare every strategy on one stock, side by side.

Runs all strategies in the registry over the same data and prints one table so
you can see how they stack up on the metrics that matter — CAGR, total return,
risk (Sharpe, drawdown) and win rate — versus simply buying and holding.

USAGE
  python compare.py --ticker AAPL
  python compare.py --ticker RELIANCE.NS --start 2018-01-01
"""

import argparse

from data import fetch_data
from strategies import STRATEGIES
from backtest import run_backtest, metrics, trade_stats


def main():
    p = argparse.ArgumentParser(description="Compare all strategies on one stock")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    args = p.parse_args()

    base = fetch_data(args.ticker, args.start, args.end)

    rows = []          # one row per strategy
    bh_row = None      # buy & hold benchmark (same for all)

    for name, fn in STRATEGIES.items():
        df = run_backtest(fn(base, fast=args.fast, slow=args.slow))
        m = metrics(df["strategy_return"], df["strat_equity"])
        ts = trade_stats(df)
        rows.append({
            "name": name,
            "CAGR": m["CAGR (per yr)"],
            "Return": m["Total return"],
            "Sharpe": m["Sharpe ratio"],
            "MaxDD": m["Max drawdown"],
            "Win%": ts["Win rate"],
            "Trades": ts["Round-trip trades"],
            "Hold": ts["Avg hold (days)"],
        })
        if bh_row is None:
            bh = metrics(df["market_return"], df["bh_equity"])
            bh_row = {
                "name": "buy_and_hold",
                "CAGR": bh["CAGR (per yr)"],
                "Return": bh["Total return"],
                "Sharpe": bh["Sharpe ratio"],
                "MaxDD": bh["Max drawdown"],
                "Win%": "—",
                "Trades": "—",
                "Hold": "—",
            }

    all_rows = [bh_row] + rows

    # Print the comparison table.
    hdr = (f"{'Strategy':<22}{'CAGR':>9}{'Return':>10}{'Sharpe':>9}"
           f"{'MaxDD':>9}{'Win%':>7}{'Trades':>8}{'Hold':>7}")
    print("\n" + "=" * len(hdr))
    print(f"  {args.ticker}   {args.start} -> {args.end}")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in all_rows:
        print(f"{r['name']:<22}{r['CAGR']:>9}{r['Return']:>10}{r['Sharpe']:>9}"
              f"{r['MaxDD']:>9}{r['Win%']:>7}{r['Trades']:>8}{r['Hold']:>7}")
    print("=" * len(hdr))
    print("  CAGR = compound annual growth rate (steady %/yr, with compounding).")
    print("  Hold = avg trading days per trade (~21 = 1 month). Higher CAGR/Sharpe")
    print("  better; smaller (less negative) MaxDD better.")


if __name__ == "__main__":
    main()
