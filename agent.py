"""
The monitoring agent — SUGGEST-ONLY by design.

What it does each time you run it:
  1. Check the strategy's recent health (drawdown, vs-benchmark, momentum).
  2. If it looks fine, say so and stop.
  3. If it's struggling, search for a better parameter set — but ONLY adopt a
     suggestion if it beats the current config OUT-OF-SAMPLE (on data the search
     never saw). Then it *recommends* the change in plain English.

It never edits your config or places trades. A human reads the suggestion and
decides. This is the disciplined version of "auto-tweak": it refuses to chase
short-term noise, and every proposal must survive an out-of-sample test —
because we showed that blind re-tuning just overfits and falls apart live.
"""

import argparse

from data import fetch_data
from strategies import STRATEGIES
from monitor import build_report, health_check, current_status
from oos import evaluate, FAST_GRID, SLOW_GRID


def diagnose(ticker: str, strategy: str, fast: int, slow: int,
             start: str, end: str, split: float = 0.7,
             min_improvement: float = 0.20) -> dict:
    """
    Returns a structured recommendation. `min_improvement` is how much better
    (fractionally, on out-of-sample Sharpe) a candidate must be before we'd
    bother suggesting a change — a buffer so we don't react to tiny noise.
    """
    df = build_report(ticker, strategy, start, end, fast=fast, slow=slow)
    flags = health_check(df)
    status = current_status(df)

    result = {
        "ticker": ticker,
        "strategy": strategy,
        "current_params": {"fast": fast, "slow": slow},
        "status": status,
        "flags": flags,
        "healthy": len(flags) == 0,
        "suggestion": None,
        "narrative": [],
    }

    if not flags:
        result["narrative"].append(
            f"✅ {ticker}/{strategy} looks healthy as of {status['asof']}. "
            "No action recommended.")
        return result

    # Unhealthy → search for a candidate, but validate out-of-sample.
    base = fetch_data(ticker, start, end)
    fn = STRATEGIES[strategy]
    cut = int(len(base) * split)

    scored = []
    for f in FAST_GRID:
        for s in SLOW_GRID:
            if f >= s:
                continue
            m = evaluate(base, fn, f, s, cut)
            scored.append({"fast": f, "slow": s, **m})

    # Pick the candidate that's best on the TRAIN window (the honest choice).
    candidate = max(scored, key=lambda r: (r["train_sharpe"]
                                           if r["train_sharpe"] == r["train_sharpe"] else -1e9))
    current = next((r for r in scored if r["fast"] == fast and r["slow"] == slow), None)
    cur_test = current["test_sharpe"] if current else float("nan")
    cand_test = candidate["test_sharpe"]

    # Build the human-readable diagnosis.
    result["narrative"].append(f"⚠️  {ticker}/{strategy} flagged {len(flags)} issue(s):")
    for sev, msg in flags:
        result["narrative"].append(f"     • [{sev}] {msg}")

    same = (candidate["fast"], candidate["slow"]) == (fast, slow)
    improved = (cand_test == cand_test) and (cur_test == cur_test) \
        and cand_test > cur_test * (1 + min_improvement) and cand_test > 0

    if same:
        result["narrative"].append(
            "  The current parameters are still the best on the training data — "
            "this looks like normal variance, not a broken strategy. "
            "RECOMMENDATION: hold; do not change anything.")
    elif improved:
        result["suggestion"] = {
            "fast": candidate["fast"], "slow": candidate["slow"],
            "current_oos_sharpe": round(cur_test, 2),
            "suggested_oos_sharpe": round(cand_test, 2),
        }
        result["narrative"].append(
            f"  A candidate fast={candidate['fast']}, slow={candidate['slow']} "
            f"beats the current config OUT-OF-SAMPLE "
            f"(test Sharpe {cand_test:.2f} vs {cur_test:.2f}).")
        result["narrative"].append(
            "  RECOMMENDATION: review this change. If you agree, re-run with "
            f"--fast {candidate['fast']} --slow {candidate['slow']} (paper only).")
    else:
        result["narrative"].append(
            "  Found alternatives, but none clear the out-of-sample bar — they win "
            "on the past but not on unseen data (classic overfit). "
            "RECOMMENDATION: hold; changing now would likely chase noise.")

    return result


def main():
    p = argparse.ArgumentParser(description="Suggest-only monitoring agent")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--strategy", default="sma_crossover", choices=list(STRATEGIES))
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2024-12-31")
    args = p.parse_args()

    rec = diagnose(args.ticker, args.strategy, args.fast, args.slow, args.start, args.end)
    print("\n" + "=" * 60)
    for line in rec["narrative"]:
        print(line)
    print("=" * 60)
    print("  (suggest-only: nothing was changed or traded)")


if __name__ == "__main__":
    main()
