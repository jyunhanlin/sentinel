# TG Message Format Unification — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify all Telegram bot messages to consistent English + emoji label style and remove all Chinese text.

**Architecture:** Pure text/UI changes across two files: `formatters.py` (message templates) and `bot.py` (buttons/prompts). No logic changes. Each formatter function gets updated to use emoji + English labels matching the `format_execution_plan()` style.

**Tech Stack:** Python, python-telegram-bot, Pydantic

---

### Task 1: Update `formatters.py` — `format_proposal()`

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/formatters.py:107-161`
- Test: `orchestrator/tests/unit/test_telegram.py` (class `TestFormatProposal`)

**Step 1: Update `format_proposal()` to use emoji labels**

Replace `\u25b6` (▶) with `\U0001f3af` (🎯) for entry. Add emoji prefixes to all data lines.

```python
def format_proposal(result: PipelineResult) -> str:
    if result.proposal is None:
        return (
            f"\u274c {result.status.upper()}\n"
            f"{result.rejection_reason or 'No proposal generated'}"
        )

    p = result.proposal
    is_flat = p.side.value == "flat"
    emoji = _SIDE_EMOJI.get(p.side.value, "")
    time_str = _fmt_time(result.created_at)

    if is_flat:
        lines = [
            f"{emoji} FLAT {p.symbol}",
            f"\U0001f4ca Confidence: {p.confidence:.0%} \u00b7 {p.time_horizon}",
            f"\n\U0001f4a1 {p.rationale}",
        ]
    elif result.status == "rejected":
        lines = [
            f"\u274c REJECTED \u2014 {p.symbol}",
            time_str,
            f"\n{result.rejection_reason}",
        ]
    else:
        entry_price = p.entry.price if p.entry.price else None
        lines = [
            f"{emoji} {p.side.value.upper()} {p.symbol}",
            time_str,
            "",
            f"\U0001f3af Entry: {p.entry.type}"
            + (f" @ ${p.entry.price:,.1f}" if p.entry.price else ""),
        ]
        if p.stop_loss is not None:
            sl_pct = ""
            if entry_price and entry_price > 0:
                pct = (p.stop_loss - entry_price) / entry_price * 100
                sl_pct = f" ({pct:+.1f}%)"
            lines.append(f"\u26d4 Stop Loss: ${p.stop_loss:,.1f}{sl_pct}")
        lines.extend(_format_tp_lines(p.take_profit, entry_price))
        lines.append(
            f"\n{p.suggested_leverage}x"
            f" \u00b7 \U0001f4ca Confidence: {p.confidence:.0%}"
            f" \u00b7 {p.time_horizon}"
        )
        lines.append(f"\n\U0001f4a1 {p.rationale}")

    if result.model_used:
        lines.append(f"\nModel: {result.model_used.split('/')[-1]}")

    degraded = _degraded_labels(result)
    if degraded:
        lines.append(f"\u26a0\ufe0f Degraded: {', '.join(degraded)}")

    return "\n".join(lines)
```

**Step 2: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py::TestFormatProposal -v`
Expected: All pass (tests check for "Stop Loss", "Take Profit", "Confidence" which are preserved)

**Step 3: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/formatters.py
git commit -m "feat: unify format_proposal() to emoji label style"
```

---

### Task 2: Update `formatters.py` — `format_pending_approval()`

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/formatters.py:168-259`
- Test: `orchestrator/tests/unit/test_telegram.py` (class `TestFormatPendingApproval`)

**Step 1: Update `format_pending_approval()` to use emoji labels**

Replace `\u25b6` (▶) with `\U0001f3af` (🎯). Add emoji to all field labels.

```python
def format_pending_approval(
    approval: PendingApproval,
    *,
    technical_short: Any | None = None,
) -> str:
    """Format a PendingApproval with detailed analysis report."""
    p = approval.proposal
    entry_price = approval.snapshot_price
    emoji = _SIDE_EMOJI.get(p.side.value, "")
    time_str = _fmt_time(approval.created_at)

    lines = [
        f"{emoji} {p.side.value.upper()} {p.symbol}",
        time_str,
        "",
        f"\U0001f3af Entry: ${entry_price:,.1f} ({p.entry.type})",
    ]

    if p.stop_loss is not None:
        sl_pct = (p.stop_loss - entry_price) / entry_price * 100
        lines.append(
            f"\u26d4 Stop Loss: ${p.stop_loss:,.1f} ({sl_pct:+.1f}%)"
        )

    for i, tp in enumerate(p.take_profit, 1):
        tp_pct = (tp.price - entry_price) / entry_price * 100
        lines.append(
            f"\u2705 Take Profit {i}: ${tp.price:,.1f}"
            f" ({tp_pct:+.1f}%) \u2192 close {tp.close_pct}%"
        )

    # Trade parameters
    lines.append("")
    lines.append(f"\U0001f4b5 Leverage: {p.suggested_leverage}x")
    if technical_short and hasattr(technical_short, "volatility_pct"):
        lines.append(f"\U0001f4c8 Volatility: {technical_short.volatility_pct:.1f}%")
    if p.stop_loss is not None and p.take_profit:
        risk_dist = abs(entry_price - p.stop_loss)
        reward_dist = abs(p.take_profit[-1].price - entry_price)
        rr = reward_dist / risk_dist if risk_dist > 0 else 0
        lines.append(f"\U0001f4ca Risk/Reward: 1:{rr:.1f}")
    lines.append(f"\U0001f4ca Confidence: {p.confidence:.0%}")
    lines.append(f"\u26a0\ufe0f Risk: {p.position_size_risk_pct}%")
    lines.append(f"\u23f0 Time Horizon: {p.time_horizon}")

    # Technical analysis section
    if technical_short:
        trend_str = (
            technical_short.trend.value.upper()
            if hasattr(technical_short.trend, "value")
            else str(technical_short.trend).upper()
        )
        vol_regime = (
            technical_short.volatility_regime.value.upper()
            if hasattr(technical_short.volatility_regime, "value")
            else str(technical_short.volatility_regime).upper()
        )
        lines.append(f"\n\U0001f4c8 Trend: {trend_str} | Volatility: {vol_regime}")

        supports = [
            kl for kl in technical_short.key_levels if kl.type == "support"
        ]
        resists = [
            kl for kl in technical_short.key_levels if kl.type == "resistance"
        ]
        if supports:
            s_str = " / ".join(f"{kl.price:,.0f}" for kl in supports)
            lines.append(f"Support: {s_str}")
        if resists:
            r_str = " / ".join(f"{kl.price:,.0f}" for kl in resists)
            lines.append(f"Resistance: {r_str}")

        if technical_short.risk_flags:
            lines.append("")
            lines.append("\u26a0\ufe0f Warnings:")
            for flag in technical_short.risk_flags:
                label = flag.replace("_", " ").title()
                lines.append(f"\u2022 {label}")

    lines.append(f"\n\U0001f4a1 {p.rationale}")

    # Footer
    footer_parts: list[str] = []
    if approval.model_used:
        footer_parts.append(f"Model: {approval.model_used.split('/')[-1]}")
    remaining = int(
        (approval.expires_at - approval.created_at).total_seconds() / 60
    )
    footer_parts.append(f"Expires in {remaining} min")
    lines.append(f"\n{' \u00b7 '.join(footer_parts)}")

    return "\n".join(lines)
```

**Step 2: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py::TestFormatPendingApproval -v`
Expected: All pass. Tests check "Leverage: 10x", "Confidence: 75%", "Risk: 1.5%", "Time Horizon: 4h", "Expires in 15 min" — all preserved.

**Step 3: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/formatters.py
git commit -m "feat: unify format_pending_approval() to emoji label style"
```

---

### Task 3: Update `formatters.py` — `format_execution_result()` and `format_trade_report()`

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/formatters.py:266-308`
- Test: `orchestrator/tests/unit/test_telegram.py` (classes `TestFormatExecutionResult`, `TestFormatTradeReport`)

**Step 1: Update both functions**

```python
def format_execution_result(result: ExecutionResult) -> str:
    """Format an ExecutionResult for TG confirmation."""
    emoji = _SIDE_EMOJI.get(result.side, "\u2705")
    lines = [
        f"{emoji} {result.side.upper()} {result.symbol}"
        f" \u00b7 {result.mode}",
        "",
        f"\U0001f3af Entry: ${result.entry_price:,.1f}",
        f"\U0001f4e6 Quantity: {result.quantity:.4f}",
        f"\U0001f4b5 Fees: ${result.fees:,.2f}",
    ]
    if result.sl_order_id:
        lines.append(f"\u26d4 SL order: {result.sl_order_id}")
    if result.tp_order_id:
        lines.append(f"\u2705 TP order: {result.tp_order_id}")
    return "\n".join(lines)


def format_trade_report(result: CloseResult) -> str:
    reason_label = {
        "sl": "SL", "tp": "TP", "liquidation": "LIQ",
        "manual": "MANUAL", "partial_reduce": "PARTIAL",
    }.get(result.reason, result.reason.upper())
    if result.partial:
        reason_label = f"PARTIAL {reason_label}"

    pnl_emoji = "\U0001f4c8" if result.pnl >= 0 else "\U0001f4c9"
    side_str = (
        result.side.value.upper()
        if hasattr(result.side, "value")
        else str(result.side).upper()
    )

    lines = [
        f"{pnl_emoji} {reason_label} \u2014 {result.symbol} {side_str}",
        "",
        f"\U0001f3af Entry: ${result.entry_price:,.1f} \u2192 Exit: ${result.exit_price:,.1f}",
        f"\U0001f4b0 PnL: {_pnl_str(result.pnl)} (fees ${result.fees:,.2f})",
    ]
    if result.remaining_quantity is not None:
        lines.append(f"\U0001f4e6 Remaining: {result.remaining_quantity:.4f}")
    return "\n".join(lines)
```

**Step 2: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py::TestFormatExecutionResult tests/unit/test_telegram.py::TestFormatTradeReport -v`
Expected: All pass. Tests check for "LONG", "live"/"paper", price strings, "SL", "TP", PnL strings — all preserved.

**Step 3: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/formatters.py
git commit -m "feat: unify format_execution_result() and format_trade_report() to emoji label style"
```

---

### Task 4: Update `formatters.py` — `format_position_card()` and `format_account_overview()`

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/formatters.py:539-611`
- Test: `orchestrator/tests/unit/test_telegram.py` (class `TestStatusWithPositions`)

**Step 1: Update both functions**

```python
def format_position_card(info: dict) -> str:
    """Format a single position with PnL info for /status display."""
    pos = info["position"]
    pnl = info["unrealized_pnl"]
    pnl_pct = info["pnl_pct"]
    roe_pct = info["roe_pct"]

    pnl_sign = "+" if pnl >= 0 else ""
    side_str = (
        pos.side.value.upper()
        if hasattr(pos.side, "value")
        else str(pos.side).upper()
    )
    leverage_str = f" {pos.leverage}x" if pos.leverage > 1 else ""

    lines = [
        f"{pos.symbol}  {side_str}{leverage_str}",
        "",
        f"\U0001f3af Entry: ${pos.entry_price:,.1f}",
        f"\U0001f4e6 Qty: {pos.quantity:.4f}",
    ]
    if pos.margin > 0:
        lines.append(
            f"\U0001f4b5 Margin: ${pos.margin:,.2f}"
            f" \u00b7 \U0001f480 Liq: ${pos.liquidation_price:,.1f}"
        )
    lines.append(f"\u26d4 Stop Loss: ${pos.stop_loss:,.1f}")
    if pos.take_profit:
        tp_str = ", ".join(
            f"${tp.price:,.1f} ({tp.close_pct}%)"
            for tp in pos.take_profit
        )
        lines.append(f"\u2705 Take Profit: {tp_str}")

    lines.append(
        f"\n\U0001f4b0 PnL: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)"
    )
    if pos.margin > 0:
        lines.append(f"\U0001f4c8 ROE: {pnl_sign}{roe_pct:.2f}%")

    return "\n".join(lines)


def format_account_overview(
    *,
    equity: float,
    available: float,
    used_margin: float,
    initial_equity: float,
    position_count: int = 0,
) -> str:
    """Format account overview with margin info."""
    total_pnl = equity - initial_equity
    pnl_sign = "+" if total_pnl >= 0 else ""

    lines = [
        "Account Overview",
        "",
        f"\U0001f4b5 Equity: ${equity:,.2f}"
        f" ({pnl_sign}${total_pnl:,.2f})",
        f"\U0001f4b0 Available: ${available:,.2f}",
        f"\U0001f4ca Used Margin: ${used_margin:,.2f}",
    ]

    lines.append(
        f"\n\U0001f4e6 Open Positions: {position_count}"
        if position_count > 0
        else "\nNo open positions"
    )

    return "\n".join(lines)
```

**Step 2: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py::TestStatusWithPositions -v`
Expected: All pass. Tests check "Account Overview", "Open Positions: 1", "BTC", "No open positions" — all preserved.

**Step 3: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/formatters.py
git commit -m "feat: unify format_position_card() and format_account_overview() to emoji label style"
```

---

### Task 5: Update `formatters.py` — `format_perf_report()` and `format_history_paginated()`

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/formatters.py:481-655`
- Test: `orchestrator/tests/unit/test_telegram.py` (classes `TestFormatPerfReport`, `TestFormatHistory`)

**Step 1: Update both functions**

```python
def format_perf_report(stats: PerformanceStats) -> str:
    if stats.total_trades == 0:
        return (
            "No trades yet. Performance report will be "
            "available after closing positions."
        )

    pnl_str = _pnl_str(stats.total_pnl)
    pnl_pct_sign = "+" if stats.total_pnl_pct >= 0 else ""
    pf_str = (
        "inf"
        if stats.profit_factor == float("inf")
        else f"{stats.profit_factor:.2f}"
    )

    lines = [
        "Performance",
        "",
        f"\U0001f4b0 PnL: {pnl_str}"
        f" ({pnl_pct_sign}{stats.total_pnl_pct:.1f}%)",
        f"\U0001f3af Win Rate: {stats.win_rate:.1%}"
        f" ({stats.winning_trades}/{stats.total_trades})",
        f"\U0001f4ca Profit Factor: {pf_str}",
        f"\U0001f4c9 Max DD: {stats.max_drawdown_pct:.1f}%",
        f"\U0001f4c8 Sharpe: {stats.sharpe_ratio:.2f}",
    ]
    return "\n".join(lines)


def format_history_paginated(
    trades: list[PaperTradeRecord],
    page: int,
    total_pages: int,
) -> str:
    """Format closed trades with pagination info."""
    if not trades:
        return "No closed trades yet."

    lines = [f"History \u2014 {page}/{total_pages}"]
    for t in trades:
        side_str = (
            t.side.upper()
            if isinstance(t.side, str)
            else t.side.value.upper()
        )
        leverage_str = f" {t.leverage}x" if t.leverage > 1 else ""
        reason_str = (
            f" ({t.close_reason.upper()})" if t.close_reason else ""
        )

        lines.append(
            f"\n  {t.symbol} {side_str}{leverage_str}{reason_str}"
        )
        lines.append(
            f"  \U0001f3af ${t.entry_price:,.1f} \u2192 ${t.exit_price:,.1f}"
            f" \u00b7 \U0001f4b0 {_pnl_str(t.pnl)}"
        )

        if t.margin > 0:
            roe = (t.pnl / t.margin * 100) if t.margin else 0
            roe_sign = "+" if roe >= 0 else ""
            lines.append(
                f"  \U0001f4b5 Margin: ${t.margin:,.2f}"
                f" \u00b7 \U0001f4c8 ROE: {roe_sign}{roe:.2f}%"
            )

    return "\n".join(lines)
```

**Step 2: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py::TestFormatPerfReport tests/unit/test_telegram.py::TestFormatHistory -v`
Expected: All pass. Tests check for PnL values, percentages, symbol names — all preserved.

**Step 3: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/formatters.py
git commit -m "feat: unify format_perf_report() and format_history_paginated() to emoji label style"
```

---

### Task 6: Update `formatters.py` — Remove `── Analysis ──` divider from `format_execution_plan()`

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/formatters.py:793-795`
- Test: `orchestrator/tests/unit/test_formatters_execution_plan.py`

**Step 1: Remove divider line**

In `format_execution_plan()`, change the analysis section header from:

```python
        lines.append("")
        lines.append("\u2500\u2500 Analysis \u2500\u2500")
        lines.append("")
```

to just a blank line (the "Analysis" label is still present via the individual field labels):

```python
        lines.append("")
```

**Step 2: Update test assertion**

In `test_formatters_execution_plan.py`, line 101, change:

```python
    assert "Analysis" in text
```

to:

```python
    assert "Technical" in text
```

The test already asserts "Technical", "Positioning", "Catalyst", "Correlation" individually, so removing the "Analysis" header assertion is sufficient.

Also update `test_format_without_analysis_shows_upper_only` — the assertion `assert "Analysis" not in text` still holds since the divider is removed, but we should keep it for safety since individual analysis labels won't appear either.

**Step 3: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_formatters_execution_plan.py -v`
Expected: All pass.

**Step 4: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/formatters.py orchestrator/tests/unit/test_formatters_execution_plan.py
git commit -m "feat: remove Analysis divider line from execution plan"
```

---

### Task 7: Update `bot.py` — Chinese buttons and prompts to English

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py` (lines 500, 1320, 1324, 1331, 1360, 1367-1368, 1391-1393, 1407-1410, 1424-1426, 1466, 2036, 2040, 2046, 2050, 2056, 2062, 2073, 2091-2094, 2110-2115, 2176)

**Step 1: Replace all Chinese text in bot.py**

All replacements (use exact unicode escapes from the source):

1. Line 500: `"\u2699\ufe0f \u7ba1\u7406"` → `"\u2699\ufe0f Manage"`
2. Line 1320: `"\U0001f680 \u78ba\u8a8d\u958b\u5009"` → `"\U0001f680 Confirm"`
3. Line 1324: `"\u274c \u53d6\u6d88"` → `"\u274c Cancel"`
4. Line 1331: `"\u270f\ufe0f \u8981\u8abf\u6574\u54ea\u500b\u53c3\u6578\uff1f"` → `"\u270f\ufe0f Which parameter to adjust?"`
5. Line 1360: `"\u2b05\ufe0f \u8fd4\u56de"` → `"\u2b05\ufe0f Back"`
6. Lines 1367-1368: `f"\u76ee\u524d Leverage: {current_lev}x\n" "\u9078\u64c7\u65b0\u7684\u500d\u6578\uff1a"` → `f"Current Leverage: {current_lev}x\n" "Select new leverage:"`
7. Lines 1391-1393: SL prompt → `"\u26d4 Enter new Stop Loss price:\n" "e.g. 92500"`
8. Lines 1407-1410: TP prompt → `"\u2705 Enter new Take Profit levels:\n" "e.g. 97000 50%, 99000 100%\n\n" "Format: price close%, " "price close%"`
9. Lines 1424-1426: margin prompt → `"\U0001f4b0 Enter new margin amount (USDT):\n" "e.g. 300"`
10. Line 1466: `"\u274c \u8abf\u6574\u5df2\u53d6\u6d88\u3002"` → `"\u274c Adjustment cancelled."`
11. Line 2036: `"\u79fb SL"` → `"Move SL"`
12. Line 2040: `"\u8abf TP"` → `"Adjust TP"`
13. Line 2046: `"\u52a0\u5009"` → `"Add"`
14. Line 2050: `"\u6e1b\u5009"` → `"Reduce"`
15. Line 2056: `"\u5e73\u5009"` → `"Close"`
16. Line 2062: `"\u2b05\ufe0f \u8fd4\u56de"` → `"\u2b05\ufe0f Back"`
17. Lines 2091-2094: pos SL prompt → `"\u26d4 Enter new Stop Loss price:\n" "e.g. 92500"`
18. Lines 2110-2115: pos TP prompt → `"\u2705 Enter new Take Profit levels:\n" "e.g. 97000 50%, 99000 100%\n\n" "Format: price close%, " "price close%"`
19. Line 2176: `"\u2699\ufe0f \u7ba1\u7406"` → `"\u2699\ufe0f Manage"`

**Step 2: Run full test suite**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py -v`
Expected: All pass. No tests assert on Chinese button text.

**Step 3: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/bot.py
git commit -m "feat: replace all Chinese button labels and prompts with English"
```

---

### Task 8: Run full test suite and verify no Chinese remains

**Step 1: Run all tests**

Run: `cd orchestrator && uv run pytest -v --cov=orchestrator`
Expected: All tests pass, coverage ≥ 80%.

**Step 2: Verify no Chinese Unicode remains**

Run: `grep -rn '[\u4e00-\u9fff]' orchestrator/src/orchestrator/telegram/` (should return nothing)
Also: `grep -rn '\\u[4-9][0-9a-f]\{3\}' orchestrator/src/orchestrator/telegram/bot.py` to catch escaped Chinese.

**Step 3: Run linter**

Run: `cd orchestrator && uv run ruff check src/ tests/`
Expected: No errors.

**Step 4: Test `/preview` command manually (optional)**

The `/preview plan` command in bot.py generates mock messages — use it to visually verify formatting.

**Step 5: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address any remaining formatting issues"
```
