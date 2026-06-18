"""
Out-of-sample (train/test) validation — the antidote to fooling yourself.

THE PROBLEM
If you try many parameter combinations and keep the one with the best
backtest, you've "overfit": you found numbers that happened to fit the past,
not a real edge. It will usually fall apart on new data.

THE TEST
1. Split history into TRAIN (early years) and TEST (recent years).
2. Grid-search parameters using ONLY the train years; lock the winner.
3. Evaluate those locked parameters on the TEST years it never saw.
4. Compare. If train looked great but test is poor → it was overfit.

We also show the BEST-possible test parameters. If the train winner isn't the
test winner (it usually isn't), that gap IS the overfitting, made visible.

USAGE
  python oos.py --ticker AAPL --strategy sma_crossover
  python oos.py --ticker RELIANCE.NS --strategy donchian_breakout --split 0.7
"""

import argparse
import numpy as np
import pandas as pd

from data import fetch_data
from strategies import STRATEGIES
from backtest import run_backtest

TRADING_DAYS = 252


def sharpe(returns: pd.Series) -> float:
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    return (returns.mean() * TRADING_DAYS) / vol if vol > 0 else float("nan")


def cagr(returns: pd.Series) -> float:
    if len(returns) == 0:
        return float("nan")
    equity = (1 + returns).cumprod().iloc[-1]
    years = len(returns) / TRADING_DAYS
    return equity ** (1 / years) - 1 if years > 0 else float("nan")


# Parameter grid to search. (fast must be < slow.)
FAST_GRID = [5, 10, 15, 20, 30]
SLOW_GRID = [20, 30, 50, 100, 200]


def evaluate(base, strat_fn, fast, slow, cut):
    """
    Run a strategy on the FULL series (so test bars get proper warmup), then
    split the daily returns into train/test. Returns metrics for each side.
    Parameter SELECTION only ever looks at the train metrics — no leakage.
    """
    full = run_backtest(strat_fn(base, fast=fast, slow=slow))
    ret = full["strategy_return"]
    return {
        "train_sharpe": sharpe(ret.iloc[:cut]),
        "train_cagr": cagr(ret.iloc[:cut]),
        "test_sharpe": sharpe(ret.iloc[cut:]),
        "test_cagr": cagr(ret.iloc[cut:]),
    }


def main():
    p = argparse.ArgumentParser(description="Out-of-sample train/test validation")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--strategy", default="sma_crossover", choices=list(STRATEGIES))
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--split", type=float, default=0.7, help="fraction used for training")
    p.add_argument("--metric", default="sharpe", choices=["sharpe", "cagr"])
    args = p.parse_args()

    base = fetch_data(args.ticker, args.start, args.end)
    strat_fn = STRATEGIES[args.strategy]
    cut = int(len(base) * args.split)
    split_date = base.index[cut].date()
    key = f"train_{args.metric}"
    test_key = f"test_{args.metric}"

    # Evaluate every combo once; record train + test metrics.
    results = []
    for f in FAST_GRID:
        for s in SLOW_GRID:
            if f >= s:
                continue
            m = evaluate(base, strat_fn, f, s, cut)
            results.append({"fast": f, "slow": s, **m})

    # The honest pick: best on TRAIN only.
    train_best = max(results, key=lambda r: (r[key] if r[key] == r[key] else -1e9))
    # The cheater's pick (for contrast): best on TEST.
    test_best = max(results, key=lambda r: (r[test_key] if r[test_key] == r[test_key] else -1e9))

    print("\n" + "=" * 64)
    print(f"  {args.ticker}  |  {args.strategy}  |  optimize for {args.metric.upper()}")
    print(f"  TRAIN: {args.start} -> {split_date}   TEST: {split_date} -> {args.end}")
    print("=" * 64)

    print("\n  STEP 1 — pick best params using ONLY the train years:")
    print(f"    winner: fast={train_best['fast']}, slow={train_best['slow']}")
    print(f"    train  Sharpe={train_best['train_sharpe']:.2f}  CAGR={train_best['train_cagr']:.1%}")

    print("\n  STEP 2 — lock those params, judge on UNSEEN test years:")
    print(f"    test   Sharpe={train_best['test_sharpe']:.2f}  CAGR={train_best['test_cagr']:.1%}")

    # Verdict
    deg = train_best[f"train_{args.metric}"] - train_best[f"test_{args.metric}"]
    print("\n  VERDICT:")
    if train_best["test_sharpe"] != train_best["test_sharpe"] or train_best["test_sharpe"] < 0:
        print("    ❌ Fell apart out-of-sample — the train result was overfit / luck.")
    elif deg > (0.5 if args.metric == "sharpe" else 0.10):
        print("    ⚠️  Degraded a lot on unseen data — treat the backtest with suspicion.")
    else:
        print("    ✅ Held up reasonably out-of-sample — more (not fully) trustworthy.")

    print("\n  OVERFITTING CHECK — was the train winner also the test winner?")
    print(f"    best on TRAIN: fast={train_best['fast']}, slow={train_best['slow']}"
          f"  (test {args.metric} = {train_best[test_key]:.2f})")
    print(f"    best on TEST : fast={test_best['fast']}, slow={test_best['slow']}"
          f"  (test {args.metric} = {test_best[test_key]:.2f})")
    if (train_best["fast"], train_best["slow"]) != (test_best["fast"], test_best["slow"]):
        print("    → They differ. The params that won the past did NOT win the future.")
        print("      That gap is overfitting. You could not have known the test winner")
        print("      in advance — which is exactly the point.")
    else:
        print("    → Same params won both — a (rare) encouraging sign.")
    print("=" * 64)


if __name__ == "__main__":
    main()
