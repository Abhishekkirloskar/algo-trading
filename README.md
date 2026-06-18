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
| `data.py` | Downloads & caches historical prices via `yfinance` (US + India) |
| `strategies.py` | The strategies (4 of them) + the live-signal helper |
| `backtest.py` | Simulates one strategy, prints metrics, draws the chart |
| `compare.py` | Runs **all** strategies side by side in one table |
| `oos.py` | **Out-of-sample** train/test validation (catches overfitting) |
| `paper_trade.py` | Live paper trading on Alpaca (with a no-keys dry-run) |
| `monitor.py` | Performance tracking: equity, drawdown, rolling Sharpe, trade log, health checks |
| `agent.py` | Suggest-only monitoring agent (flags trouble, proposes OOS-validated changes) |
| `dashboard.py` | Live Streamlit dashboard tying it all together |

## The four strategies

| Name | Type | Idea | Profile |
|---|---|---|---|
| `sma_crossover` | trend | long when fast SMA > slow SMA | few big wins, many small losses |
| `sma_crossover_trend` | trend | crossover, but only above the 200-day | fewer whipsaws, smaller drawdown |
| `rsi_mean_reversion` | mean-reversion | buy oversold dips in an uptrend (Connors RSI-2) | high win rate, small edge, many trades |
| `donchian_breakout` | breakout/swing | buy N-day highs, exit on M-day lows | classic swing/trend |

```bash
python backtest.py --ticker AAPL --strategy donchian_breakout
python compare.py  --ticker AAPL                 # all strategies at once
```

Tune the horizon with `--fast`/`--slow`: smaller = shorter holds (swing,
days–weeks); larger = longer holds (position, months). The `Hold` column in
`compare.py` shows the average days held so you can target a horizon.

## Reading the results

- **CAGR** — compound annual growth rate: the steady %/year, *with* compounding.
  The number that reflects what actually ends up in your account.
- **Total return** — the whole-period gain (hides the time dimension; use CAGR
  to compare across periods).
- **Sharpe ratio** — return per unit of risk. >1 good, >2 great.
- **Max drawdown** — worst peak-to-trough drop. Often a strategy's real value is
  *reducing* this, even if it earns less than buy & hold.
- **Win rate** — % of trades that were profitable. **Misleading on its own** —
  read it with avg win/avg loss. You can win 40% of the time and still profit if
  wins are bigger; or win 60% and lose money (see `rsi_mean_reversion` on AAPL).
- **Strategy vs. Buy & Hold** — the honest benchmark. Beating it consistently is
  genuinely hard; that's the whole lesson.

## Validate before you trust: `oos.py`

The #1 way beginners fool themselves is **overfitting** — trying many parameters
and keeping whichever fit the past best. It won't survive live.

```bash
python oos.py --ticker AAPL --strategy sma_crossover
```

It tunes parameters on the early "train" years, locks them, then judges on the
recent "test" years it never saw. In our runs, backtest Sharpe **roughly halved**
out-of-sample *every time* (1.49 → 0.71, etc.). Lesson: **mentally discount any
backtest by ~50%**, and never trust a result you haven't tested out-of-sample.

> This is also why this repo does **not** ship "optimized" parameters: the
> best-on-history settings are usually overfit. Pick sensible round numbers,
> validate out-of-sample, and stay skeptical.

## Monitoring dashboard + the agent

A live dashboard to watch a strategy, plus a self-monitoring agent.

```bash
streamlit run dashboard.py        # opens in your browser
```

The dashboard shows headline metrics, the equity curve vs. buy & hold, drawdown
and rolling-Sharpe charts, the current position, a trade log, and the agent's
verdict — all interactive via the sidebar (ticker, strategy, parameters).

### The agent is **suggest-only** (on purpose)

You asked for something that "tweaks the algorithm if it's not performing well."
The honest version of that is *not* auto-re-tuning — that just chases noise and
overfits (we proved backtests halve out-of-sample). So the agent:

- **Checks health** with simple, transparent rules (deep drawdown, lagging the
  benchmark over ~6 months, negative 3-month rolling Sharpe).
- If struggling, **searches for a better config but only suggests it if it beats
  the current one OUT-OF-SAMPLE** (on data the search never saw). Otherwise it
  explicitly says "hold — this is variance, not a broken strategy."
- **Never edits settings or trades.** A human reviews and decides.

```bash
python agent.py --ticker AAPL --strategy sma_crossover --fast 20 --slow 50
```

This keeps the *spirit* of a self-improving system (watches itself, adapts) with
discipline instead of a self-destruct button.

## Concepts baked into the code (worth knowing)

- **Look-ahead bias** — we `shift()` signals by one day so we only act on
  information we'd actually have had. Skipping this fakes great results.
- **Trading costs** — each trade subtracts a small fee (brokerage + slippage),
  because frequent trading quietly eats returns (it sank the 200-trade RSI bot).
- **Expectancy > win rate** — `(win% × avg win) − (loss% × avg loss) − costs` is
  what actually matters.

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

1. **Walk-forward analysis** — repeat the `oos.py` train/test split across many
   rolling windows for a more robust picture than a single split.
2. Add your own strategy in `strategies.py` and register it in `STRATEGIES`.
3. Backtest a **basket** of stocks and average the results (one stock can flatter
   or fool you).
4. Automate `paper_trade.py` with `cron` to run daily after the US close.
5. Add **position sizing & risk limits** (never risk more than ~1–2% per trade).
6. Add a **benchmark index** (SPY / NIFTY) so "beat the market" is explicit.

## ⚠️ Reality check

This is an educational sandbox. Backtests are optimistic, real fills have
slippage, and markets change. Most retail algo strategies do **not** beat buying
and holding an index after costs. Treat this as a way to learn the *process*
(data → strategy → honest validation → paper trading), not a money printer.
Never risk money you can't afford to lose.
