"""
Paper trading with Alpaca (US stocks).

PAPER TRADING = trading with fake money against the REAL live market. It's the
step between backtesting (on the past) and risking actual cash. Alpaca gives
you a free, unlimited paper account with $100k of pretend money.

What this script does, once per run:
  1. Look at recent prices for a stock
  2. Ask the strategy: should we be holding it, or in cash?
  3. Compare with what we currently hold in the paper account
  4. Place a paper BUY or SELL order to match the target (if needed)

This is meant to be run ONCE per day (e.g. via cron) for a daily strategy —
not in a fast loop. Run it as many times as you like; it only trades when the
target position differs from what you already hold.

USAGE
  # Test the logic with NO account/keys needed — just prints the decision:
  python paper_trade.py --ticker AAPL --dry-run

  # Real paper trades (after you add Alpaca keys to a .env file — see README):
  python paper_trade.py --ticker AAPL
  python paper_trade.py --ticker MSFT --qty 10 --fast 20 --slow 50
"""

import os
import argparse
from datetime import date, timedelta

from dotenv import load_dotenv

from data import fetch_data
from strategies import latest_signal

load_dotenv()  # read ALPACA_* keys from a local .env file if present


def get_signal(ticker: str, fast: int, slow: int) -> dict:
    """Pull ~1 year of recent data and compute the current target position."""
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=400)).isoformat()
    df = fetch_data(ticker, start, end, use_cache=False)  # always fresh for live
    if len(df) < slow + 1:
        raise ValueError(f"Not enough data for a {slow}-day average.")
    return latest_signal(df, fast=fast, slow=slow)


def print_decision(ticker, sig, qty):
    arrow = "HOLD STOCK (long)" if sig["want_long"] else "HOLD CASH (flat)"
    print("\n" + "=" * 50)
    print(f"  {ticker}  — signal as of {sig['asof']}")
    print("=" * 50)
    print(f"  Close price : {sig['close']:.2f}")
    print(f"  Fast SMA    : {sig['sma_fast']:.2f}")
    print(f"  Slow SMA    : {sig['sma_slow']:.2f}")
    print(f"  Target      : {arrow}")
    print(f"  Target qty  : {qty if sig['want_long'] else 0} shares")
    print("=" * 50)


def run_live(ticker, sig, qty):
    """Connect to Alpaca paper account and reconcile our position to target."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    api_key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        print("\n⚠️  No Alpaca keys found. Add them to a .env file (see README),")
        print("    or run with --dry-run to just see the decision.")
        return

    # paper=True points at the paper-trading endpoint — never your real money.
    client = TradingClient(api_key, secret, paper=True)

    acct = client.get_account()
    print(f"\n  Paper account: ${float(acct.cash):,.2f} cash, "
          f"${float(acct.portfolio_value):,.2f} total")

    # How many shares of this ticker do we currently hold?
    held = 0
    for pos in client.get_all_positions():
        if pos.symbol == ticker:
            held = int(float(pos.qty))
            break

    target = qty if sig["want_long"] else 0
    diff = target - held
    print(f"  Currently held: {held}  ->  target: {target}  (diff {diff:+d})")

    if diff == 0:
        print("  ✓ Already at target — no order needed.")
        return

    side = OrderSide.BUY if diff > 0 else OrderSide.SELL
    order = MarketOrderRequest(
        symbol=ticker,
        qty=abs(diff),
        side=side,
        time_in_force=TimeInForce.DAY,
    )
    submitted = client.submit_order(order)
    print(f"  → Submitted {side.value.upper()} {abs(diff)} {ticker} "
          f"(order id {submitted.id})")
    print("  Note: market orders fill when the US market is open.")


def main():
    p = argparse.ArgumentParser(description="Alpaca paper trading (US stocks)")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--qty", type=int, default=10, help="shares to hold when long")
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--dry-run", action="store_true",
                   help="just print the decision; don't connect to Alpaca")
    args = p.parse_args()

    sig = get_signal(args.ticker, args.fast, args.slow)
    print_decision(args.ticker, sig, args.qty)

    if args.dry_run:
        print("  (dry-run: no orders placed)")
    else:
        run_live(args.ticker, sig, args.qty)


if __name__ == "__main__":
    main()
