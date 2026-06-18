"""
Live visual dashboard for monitoring a strategy.

Run it with:
    streamlit run dashboard.py

Then pick a ticker / strategy / parameters in the sidebar. You get:
  * headline metrics (CAGR, Sharpe, max drawdown, win rate)
  * equity curve vs. buy & hold
  * drawdown and rolling-Sharpe charts (how it's doing recently)
  * a trade log
  * the monitoring AGENT's health check + suggestion (suggest-only)

Everything is simulated on historical daily data — no broker keys needed.
"""

import pandas as pd
import streamlit as st

from strategies import STRATEGIES
from monitor import (build_report, trade_log, current_status,
                     summary_metrics, health_check)
from agent import diagnose

st.set_page_config(page_title="Algo Monitor", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def load(ticker, strategy, start, end, fast, slow):
    return build_report(ticker, strategy, start, end, fast=fast, slow=slow)


# ---- Sidebar controls -------------------------------------------------------
st.sidebar.title("📈 Algo Monitor")
ticker = st.sidebar.text_input("Ticker (US: AAPL · India: RELIANCE.NS)", "AAPL")
strategy = st.sidebar.selectbox("Strategy", list(STRATEGIES))
fast = st.sidebar.slider("Fast / short window", 3, 60, 20)
slow = st.sidebar.slider("Slow / long window", 10, 250, 50)
start = st.sidebar.text_input("Start", "2015-01-01")
end = st.sidebar.text_input("End", "2024-12-31")

st.title(f"{ticker} — {strategy}")
st.caption(f"fast={fast}, slow={slow} · simulated on daily data · educational only")

try:
    df = load(ticker, strategy, start, end, fast, slow)
except Exception as e:  # noqa: BLE001
    st.error(f"Couldn't load data: {e}")
    st.stop()

status = current_status(df)
sm = summary_metrics(df)

# ---- Headline metrics -------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("CAGR", sm["strategy"]["CAGR (per yr)"], help="vs buy & hold " + sm["buy_hold"]["CAGR (per yr)"])
c2.metric("Total return", sm["strategy"]["Total return"], help="vs buy & hold " + sm["buy_hold"]["Total return"])
c3.metric("Sharpe", sm["strategy"]["Sharpe ratio"], help="vs buy & hold " + sm["buy_hold"]["Sharpe ratio"])
c4.metric("Max drawdown", sm["strategy"]["Max drawdown"])
c5.metric("Win rate", sm["trades"]["Win rate"], help=f"{sm['trades']['Round-trip trades']} trades")

pos = "🟢 IN MARKET (holding)" if status["in_market"] else "⚪ IN CASH (flat)"
st.info(f"**As of {status['asof']}:** {pos} · price {status['price']:.2f} · "
        f"current drawdown {status['drawdown']:.1%}")

# ---- Charts -----------------------------------------------------------------
st.subheader("Growth of $1 — strategy vs. buy & hold")
eq = df[["strat_equity", "bh_equity"]].rename(
    columns={"strat_equity": "Strategy", "bh_equity": "Buy & Hold"})
st.line_chart(eq, height=320)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Drawdown")
    dd = df[["drawdown", "bh_drawdown"]].rename(
        columns={"drawdown": "Strategy", "bh_drawdown": "Buy & Hold"})
    st.area_chart(dd, height=260)
with col_b:
    st.subheader("Rolling 3-month Sharpe")
    st.line_chart(df[["roll_sharpe"]].rename(columns={"roll_sharpe": "Rolling Sharpe"}),
                  height=260)

# ---- Monitoring agent -------------------------------------------------------
st.subheader("🤖 Monitoring agent (suggest-only)")
rec = diagnose(ticker, strategy, fast, slow, start, end)
if rec["healthy"]:
    st.success("\n\n".join(rec["narrative"]))
else:
    st.warning("\n\n".join(rec["narrative"]))
    if rec["suggestion"]:
        s = rec["suggestion"]
        st.markdown(
            f"**Proposed change:** `fast={s['fast']}, slow={s['slow']}` "
            f"— out-of-sample Sharpe **{s['suggested_oos_sharpe']}** "
            f"vs current **{s['current_oos_sharpe']}**. "
            "Apply it yourself in the sidebar if you agree.")
st.caption("The agent never changes settings or places trades — you decide.")

# ---- Trade log --------------------------------------------------------------
st.subheader("Recent trades")
tl = trade_log(df)
if tl.empty:
    st.write("No completed round-trip trades in this window.")
else:
    tl = tl.copy()
    tl["return"] = (tl["return"] * 100).round(1).astype(str) + "%"
    st.dataframe(tl.tail(15).iloc[::-1], use_container_width=True, hide_index=True)
