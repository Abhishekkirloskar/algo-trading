# Algo Trading — Backtesting Starter

A small, well-commented Python project to learn algorithmic trading by
**backtesting**: running a strategy on historical data to see how it *would*
have performed — with zero real money at risk.

Works for **US stocks** (`AAPL`) and **Indian stocks** (`RELIANCE.NS`) using
free Yahoo Finance data.

> ⚠️ Educational only. Nothing here places real orders. Past performance does
> not predict future results, and most simple strategies do **not** beat just
> buying and holding (you'll see that below).

## Setup

```bash
cd algo-trading
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python backtest.py                              # AAPL, default settings
python backtest.py --ticker MSFT                # another US stock
python backtest.py --ticker RELIANCE.NS         # Indian stock (NSE)
python backtest.py --ticker TCS.NS --fast 10 --slow 30
python backtest.py --ticker INFY.NS --start 2018-01-01 --end 2024-12-31
```

Each run prints a metrics table and saves `backtest_result.png` (price with
buy/sell markers, plus the equity curve vs. buy & hold).

**Ticker format:** US = plain (`AAPL`, `TSLA`). India = add `.NS` for NSE or
`.BO` for BSE (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`).

## How it works

| File | Role |
|---|---|
| `data.py` | Downloads & caches historical prices via `yfinance` |
| `strategies.py` | Defines strategies (currently a moving-average crossover) |
| `backtest.py` | Simulates the strategy, computes metrics, draws the chart |

The default strategy is a **SMA crossover**: be invested when the 20-day moving
average is above the 50-day (uptrend), otherwise sit in cash.

## Reading the results

- **Total return / CAGR** — how much you'd have made (CAGR = annualized).
- **Sharpe ratio** — return per unit of risk. >1 good, >2 great.
- **Max drawdown** — the worst peak-to-trough drop. The strategy's main job is
  often to *reduce* this (less pain) even if it earns less.
- **Strategy vs. Buy & Hold** — the honest benchmark. Beating it consistently
  is genuinely hard; that's the whole lesson.

## Concepts baked into the code (worth knowing)

- **Look-ahead bias** — we `shift()` signals by one day so we only act on
  information we'd actually have had. Skipping this fakes great results.
- **Trading costs** — each trade subtracts a small fee (brokerage + slippage),
  because frequent trading quietly eats returns.

## Paper trading (US stocks via Alpaca)

**Paper trading** = trading fake money against the *real live* market. It's the
step between backtesting and risking real cash. `paper_trade.py` runs the
strategy on today's data and places paper orders on a free Alpaca account.

### Try it with no account first (dry-run)

```bash
python paper_trade.py --ticker AAPL --dry-run
```

This just prints the current decision (hold stock vs. cash) — no keys needed.

### Get free Alpaca paper keys (one-time, ~3 min)

1. Sign up at <https://alpaca.markets> (free; no funding required for paper).
2. In the dashboard, switch to **Paper Trading** (toggle, top-left).
3. Under **Home / API Keys**, click **Generate New Keys**.
4. Copy the **Key ID** and **Secret Key** (the secret shows only once).
5. In this folder:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and paste your keys. (`.env` is gitignored — keys stay private.)

### Run real paper trades

```bash
python paper_trade.py --ticker AAPL                 # hold 10 shares when long
python paper_trade.py --ticker MSFT --qty 5
```

It checks your current paper position and only trades if it differs from the
target. Run it **once a day** (orders fill when the US market is open, 9:30am–
4:00pm ET). You can automate it with `cron` later.

> India: there's no free paper-trading API. Zerodha **Kite Connect** (~₹500/mo
> + funded account) or Angel **SmartAPI** are the options — keep using the
> backtester for Indian stocks in the meantime.

## Next steps (ideas to extend)

1. Add more strategies in `strategies.py` (RSI mean-reversion, breakout, etc.)
   and register them in `STRATEGIES`.
2. Backtest a basket of stocks and compare.
3. Automate `paper_trade.py` with `cron` to run daily after the close.
4. Add position sizing & risk limits (never risk more than X% per trade).
