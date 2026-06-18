"""
Live visual dashboard for monitoring a strategy.

Run it with:
    streamlit run dashboard.py

Pick a ticker / strategy / parameters in the sidebar, tune the health-alert
thresholds, and (optionally) get an AI-written analyst note. Everything is
simulated on historical daily data — no broker keys needed. The AI analyst
needs an Anthropic API key (ANTHROPIC_API_KEY in your .env); without it, the
rule-based diagnosis is shown instead.
"""

import matplotlib.pyplot as plt
import streamlit as st

from strategies import STRATEGIES
from monitor import (build_report, trade_log, current_status,
                     summary_metrics, health_check, monthly_returns)
from agent import diagnose
import analyst

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

st.sidebar.markdown("---")
st.sidebar.subheader("Health alert thresholds")
dd_limit = st.sidebar.slider("Max drawdown before alert", -0.60, -0.05, -0.20, 0.01,
                             help="Flag when current drawdown is worse than this")
underperf = st.sidebar.slider("Lag vs buy & hold (6 mo)", 0.0, 0.30, 0.05, 0.01,
                              help="Flag when the strategy trails buy & hold by more than this over ~6 months")
sharpe_floor = st.sidebar.slider("Rolling-Sharpe floor", -1.0, 1.0, 0.0, 0.1,
                                 help="Flag when 3-month rolling Sharpe drops below this")

st.title(f"{ticker} — {strategy}")
st.caption(f"fast={fast}, slow={slow} · simulated on daily data · educational only")

try:
    df = load(ticker, strategy, start, end, fast, slow)
except Exception as e:  # noqa: BLE001
    st.error(f"Couldn't load data: {e}")
    st.stop()

status = current_status(df)
sm = summary_metrics(df)
flags = health_check(df, dd_limit=dd_limit, underperf_margin=underperf,
                     sharpe_floor=sharpe_floor)

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

# Price chart with moving averages + buy/sell markers
st.subheader("Price & trade signals")
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(df.index, df["Close"], color="black", linewidth=1, label="Close")
if "sma_fast" in df:
    ax.plot(df.index, df["sma_fast"], color="tab:blue", linewidth=1, label="Fast")
if "sma_slow" in df:
    ax.plot(df.index, df["sma_slow"], color="tab:orange", linewidth=1, label="Slow")
buys = df[(df["trade"] == 1) & (df["position"] == 1)]
sells = df[(df["trade"] == 1) & (df["position"] == 0)]
ax.scatter(buys.index, buys["Close"], marker="^", color="green", s=40, label="Buy", zorder=5)
ax.scatter(sells.index, sells["Close"], marker="v", color="red", s=40, label="Sell", zorder=5)
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
st.pyplot(fig)

col_c, col_d = st.columns(2)
with col_c:
    st.subheader("Monthly returns (%)")
    mr = monthly_returns(df)
    st.dataframe(
        mr.style.background_gradient(cmap="RdYlGn", vmin=-0.1, vmax=0.1)
          .format("{:.1%}", na_rep=""),
        use_container_width=True,
    )
with col_d:
    st.subheader("Trade return distribution")
    tl_all = trade_log(df)
    if tl_all.empty:
        st.write("No completed trades in this window.")
    else:
        fig2, ax2 = plt.subplots(figsize=(5, 3.2))
        ax2.hist(tl_all["return"] * 100, bins=20, color="tab:green",
                 edgecolor="black", alpha=0.7)
        ax2.axvline(0, color="red", linestyle="--", linewidth=1)
        ax2.set_xlabel("Trade return (%)")
        ax2.set_ylabel("Count")
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

# ---- Monitoring agent -------------------------------------------------------
st.subheader("🤖 Monitoring agent (suggest-only)")
rec = diagnose(ticker, strategy, fast, slow, start, end,
               dd_limit=dd_limit, underperf_margin=underperf, sharpe_floor=sharpe_floor)
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

# ---- AI analyst (optional) --------------------------------------------------
st.subheader("🧠 AI analyst note")
if not analyst.available():
    st.info("Add an `ANTHROPIC_API_KEY` to your `.env` to enable an AI-written "
            "analyst note (uses Claude). Showing rule-based diagnosis above.")
else:
    if st.button("Generate analyst note"):
        with st.spinner("Asking Claude…"):
            ctx = analyst.build_context(
                ticker, strategy, {"fast": fast, "slow": slow}, sm, status, flags)
            note = analyst.analyze(ctx)
        st.write(note or "(no response)")
    st.caption("AI-written, educational only — not financial advice.")

# ---- Trade log --------------------------------------------------------------
st.subheader("Recent trades")
tl = trade_log(df)
if tl.empty:
    st.write("No completed round-trip trades in this window.")
else:
    tl = tl.copy()
    tl["return"] = (tl["return"] * 100).round(1).astype(str) + "%"
    st.dataframe(tl.tail(15).iloc[::-1], use_container_width=True, hide_index=True)
