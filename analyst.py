"""
Optional LLM "analyst" — turns the numbers + the rule-based diagnosis into a
plain-English readout using Claude.

This is purely explanatory. It does NOT decide trades or change parameters —
the suggest-only agent (agent.py) already does the disciplined analysis; this
just writes it up in richer prose for a human to read.

Requires an Anthropic API key in the environment (ANTHROPIC_API_KEY), e.g. in
your .env file. If no key is present, callers fall back to the rule-based text.
"""

from __future__ import annotations  # allow "str | None" hints on Python 3.9

import os
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-4-8"

SYSTEM = """You are a quantitative trading analyst writing a short, plain-English
note for someone LEARNING algo trading. You are given a strategy's backtest
metrics, a per-trade breakdown, and an automated health diagnosis.

Write 120-180 words. Be concrete and reference the actual numbers. Cover:
1. How the strategy did vs. buy & hold, in risk-adjusted terms (not just return).
2. What the win rate + average win/loss imply about its *style* (trend vs. mean
   reversion), and whether win rate is misleading here.
3. Your read on the health flags — is this normal variance or a real problem?

Rules: This is educational, not financial advice. Never promise future returns.
Be honest when a strategy underperforms. Do not recommend changing parameters
reactively (that risks overfitting). No markdown headers; 1-2 short paragraphs."""


def available() -> bool:
    """True if an API key is configured (so the UI can show/hide the feature)."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def analyze(context: str) -> str | None:
    """
    Send the assembled context to Claude and return the analyst note.
    Returns None if unavailable or on error (caller falls back gracefully).
    """
    if not available():
        return None
    try:
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            thinking={"type": "adaptive"},  # let the model reason as needed
            system=SYSTEM,
            messages=[{"role": "user", "content": context}],
        )
        # Adaptive thinking returns thinking blocks too — keep only the text.
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text.strip() or None
    except Exception as e:  # noqa: BLE001 — surface, don't crash the dashboard
        return f"(AI analyst unavailable: {e})"


def build_context(ticker, strategy, params, summary, status, flags) -> str:
    """Assemble a compact, factual context string for the model."""
    s, bh, ts = summary["strategy"], summary["buy_hold"], summary["trades"]
    lines = [
        f"Ticker: {ticker}",
        f"Strategy: {strategy} (fast={params['fast']}, slow={params['slow']})",
        f"As of: {status['asof']} — currently {'in the market' if status['in_market'] else 'in cash'}",
        "",
        "Strategy vs Buy & Hold:",
        f"  CAGR:        {s['CAGR (per yr)']}   vs {bh['CAGR (per yr)']}",
        f"  Total return:{s['Total return']}   vs {bh['Total return']}",
        f"  Sharpe:      {s['Sharpe ratio']}   vs {bh['Sharpe ratio']}",
        f"  Max drawdown:{s['Max drawdown']}   vs {bh['Max drawdown']}",
        "",
        "Trade breakdown:",
        f"  Round trips: {ts['Round-trip trades']}",
        f"  Win rate:    {ts['Win rate']}",
        f"  Avg win:     {ts['Avg win']}   Avg loss: {ts['Avg loss']}",
        f"  Avg hold:    {ts['Avg hold (days)']} trading days",
        "",
        "Automated health flags:",
    ]
    if flags:
        lines += [f"  - [{sev}] {msg}" for sev, msg in flags]
    else:
        lines.append("  - none (looks healthy)")
    return "\n".join(lines)
