"""
Data loading for the backtester.

We use `yfinance`, a free library that pulls historical price data straight
from Yahoo Finance. The nice part: it works for BOTH US and Indian stocks.

  US stocks:     plain ticker          ->  "AAPL", "MSFT", "TSLA"
  Indian (NSE):  add a ".NS" suffix    ->  "RELIANCE.NS", "TCS.NS", "INFY.NS"
  Indian (BSE):  add a ".BO" suffix    ->  "RELIANCE.BO"

Data is cached to a local CSV the first time so re-running a backtest is
instant and doesn't hammer Yahoo.
"""

import os
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")


def fetch_data(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Download daily OHLCV (Open, High, Low, Close, Volume) data for `ticker`
    between `start` and `end` (YYYY-MM-DD strings).

    Returns a DataFrame indexed by date with at least a 'Close' column.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = ticker.replace(".", "_")
    cache_path = os.path.join(CACHE_DIR, f"{safe}_{start}_{end}.csv")

    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if not df.empty:
            return df

    print(f"Downloading {ticker} from {start} to {end} ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(
            f"No data returned for '{ticker}'. Check the symbol — Indian stocks "
            f"need a .NS (NSE) or .BO (BSE) suffix, e.g. 'RELIANCE.NS'."
        )

    # yfinance can return multi-level columns; flatten to simple names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.to_csv(cache_path)
    return df
