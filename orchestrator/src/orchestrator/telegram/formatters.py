from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from orchestrator.exchange.paper_engine import CloseResult
from orchestrator.risk.checker import RiskResult
from orchestrator.stats.calculator import PerformanceStats

if TYPE_CHECKING:
    from orchestrator.approval.manager import PendingApproval
    from orchestrator.execution.executor import ExecutionResult
    from orchestrator.pipeline.runner import PipelineResult
    from orchestrator.storage.models import PaperTradeRecord, TradeProposalRecord


def _fmt_time(dt: datetime) -> str:
    local_dt = dt.astimezone()  # convert to system local timezone
    return local_dt.strftime("%Y/%m/%d %H:%M")


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _degraded_labels(result: PipelineResult) -> list[str]:
    labels: list[str] = []
    if result.technical_short_degraded:
        labels.append("technical_short")
    if result.technical_long_degraded:
        labels.append("technical_long")
    if result.positioning_degraded:
        labels.append("positioning")
    if result.catalyst_degraded:
        labels.append("catalyst")
    if result.correlation_degraded:
        labels.append("correlation")
    if result.proposer_degraded:
        labels.append("proposer")
    return labels


_SIDE_EMOJI = {
    "long": "\U0001f7e2",
    "short": "\U0001f534",
    "flat": "\u26aa",
}


def _pnl_str(pnl: float) -> str:
    sign = "+" if pnl >= 0 else "-"
    return f"{sign}${abs(pnl):,.2f}"


# ---------------------------------------------------------------------------
# Static messages
# ---------------------------------------------------------------------------

def format_welcome() -> str:
    return (
        "Welcome to Sentinel!\n\n"
        "I analyze crypto markets using multiple AI models and generate "
        "trade proposals with risk management.\n\n"
        "Use /help to see available commands."
    )


def format_help() -> str:
    return (
        "Commands\n\n"
        "/status — Account overview & positions\n"
        "/coin <symbol> — Detailed analysis\n"
        "/run [symbol] [model] — Trigger pipeline\n"
        "/history — Trade records\n"
        "/perf — Performance report\n"
        "/eval — Run LLM evaluation\n"
        "/resume — Un-pause after risk pause\n"
        "/help — This message"
    )


# ---------------------------------------------------------------------------
# Proposal (pipeline result notification)
# ---------------------------------------------------------------------------

def _format_tp_lines(
    take_profit: list, entry_price: float | None = None,
) -> list[str]:
    """Format take-profit levels with optional % from entry."""
    lines: list[str] = []
    for i, tp in enumerate(take_profit, 1):
        price = tp.price if hasattr(tp, "price") else tp
        close_pct = tp.close_pct if hasattr(tp, "close_pct") else 100
        pct_str = ""
        if entry_price and entry_price > 0:
            pct = (price - entry_price) / entry_price * 100
            pct_str = f" ({pct:+.1f}%)"
        lines.append(
            f"\u2705 Take Profit {i}: ${price:,.1f}{pct_str}"
            f" \u2192 close {close_pct}%"
        )
    return lines


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
            f"\u274c REJECTED — {p.symbol}",
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


# ---------------------------------------------------------------------------
# Pending approval (detailed report for decision-making)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Trade close report
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Risk rejection
# ---------------------------------------------------------------------------

def format_risk_rejection(
    *, symbol: str, side: str, entry_price: float,
    risk_result: RiskResult,
) -> str:
    label = (
        "\u23f8 RISK PAUSED"
        if risk_result.action == "pause"
        else "\U0001f6ab RISK REJECTED"
    )
    lines = [
        f"{label}",
        f"{symbol} {side} @ ${entry_price:,.1f}",
        f"\n{risk_result.rule_violated}: {risk_result.reason}",
    ]
    return "\n".join(lines)


def format_risk_pause(
    *, symbol: str, side: str, entry_price: float,
    risk_result: RiskResult,
) -> str:
    """Format a risk pause notification with clear instructions."""
    lines = [
        "\u23f8 TRADING PAUSED",
        "",
        f"{symbol} {side} @ ${entry_price:,.1f}",
        "",
        f"Rule: {risk_result.rule_violated}",
        f"Reason: {risk_result.reason}",
        "",
        "All new trades are blocked until you resume.",
        "Tap the button below or use /resume to continue.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Status / overview
# ---------------------------------------------------------------------------

def format_status(results: list[PipelineResult]) -> str:
    if not results:
        return "No pipeline results yet. Use /run to trigger analysis."

    blocks = ["Pipeline Status"]
    for r in results:
        side = r.proposal.side.value.upper() if r.proposal else "N/A"
        time_str = _fmt_time(r.created_at)
        blocks.append(
            f"\n{r.symbol} — {side} [{r.status}]  ({time_str})"
        )

        if r.proposal is None:
            reason = r.rejection_reason or "No proposal generated"
            blocks.append(f"  {reason}")
        elif r.proposal.side.value == "flat":
            blocks.append(f"  Confidence: {r.proposal.confidence:.0%}")
            blocks.append(
                f"  Rationale: {_truncate(r.proposal.rationale, 80)}"
            )
        else:
            p = r.proposal
            blocks.append(
                f"  Entry: {p.entry.type}"
                f" | Risk: {p.position_size_risk_pct}%"
            )
            sl_str = (
                f"${p.stop_loss:,.1f}"
                if p.stop_loss is not None else "\u2014"
            )
            tp_str = (
                ", ".join(f"${tp.price:,.1f}" for tp in p.take_profit)
                if p.take_profit else "\u2014"
            )
            blocks.append(
                f"  Stop Loss: {sl_str} | Take Profit: {tp_str}"
            )
            blocks.append(
                f"  Confidence: {p.confidence:.0%} | {p.time_horizon}"
            )

        if r.model_used:
            blocks.append(
                f"  Model: {r.model_used.split('/')[-1]}"
            )

        degraded = _degraded_labels(r)
        if degraded:
            blocks.append(
                f"  \u26a0\ufe0f Degraded: {', '.join(degraded)}"
            )

    return "\n".join(blocks)


def format_status_from_records(
    records: list[TradeProposalRecord],
) -> str:
    """Format status from DB TradeProposalRecords."""
    import json

    if not records:
        return "No pipeline results yet. Use /run to trigger analysis."

    blocks = ["Pipeline Status (DB)"]
    for r in records:
        try:
            p = json.loads(r.proposal_json)
            symbol = p.get("symbol", "?")
            side = p.get("side", "?").upper()
            status = r.risk_check_result or "unknown"
            time_str = _fmt_time(r.created_at)
            blocks.append(
                f"\n{symbol} — {side} [{status}]  ({time_str})"
            )

            confidence = p.get("confidence", 0)
            conf_str = (
                f"{confidence:.0%}"
                if isinstance(confidence, float)
                else str(confidence)
            )

            if side == "FLAT":
                blocks.append(f"  Confidence: {conf_str}")
                rationale = p.get("rationale", "")
                if rationale:
                    blocks.append(
                        f"  {_truncate(rationale, 80)}"
                    )
            else:
                entry_type = p.get("entry", {}).get("type", "?")
                risk_pct = p.get("position_size_risk_pct", "?")
                blocks.append(
                    f"  Entry: {entry_type} | Risk: {risk_pct}%"
                )
                sl = p.get("stop_loss")
                sl_str = f"${sl:,.1f}" if sl is not None else "\u2014"
                tps = p.get("take_profit", [])
                tp_parts: list[str] = []
                for tp in tps:
                    if isinstance(tp, dict):
                        tp_parts.append(f"${tp['price']:,.1f}")
                    else:
                        tp_parts.append(f"${tp:,.1f}")
                tp_str = (
                    ", ".join(tp_parts) if tp_parts else "\u2014"
                )
                blocks.append(
                    f"  Stop Loss: {sl_str} | Take Profit: {tp_str}"
                )
                horizon = p.get("time_horizon", "?")
                blocks.append(f"  Confidence: {conf_str} | {horizon}")

        except (json.JSONDecodeError, AttributeError):
            blocks.append(
                f"\n[parse error] proposal_id={r.proposal_id}"
            )

    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Performance report
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Eval report
# ---------------------------------------------------------------------------

def format_eval_report(report: dict[str, Any]) -> str:
    lines = [
        f"Eval: {report['dataset_name']}",
        "",
        f"Cases: {report['total_cases']}"
        f" | Pass: {report['passed_cases']}"
        f" | Fail: {report['failed_cases']}",
        f"Accuracy: {report['accuracy']:.0%}",
    ]
    if report.get("consistency_score") is not None:
        lines.append(
            f"Consistency: {report['consistency_score']:.1%}"
        )
    failures = report.get("failures", [])
    if failures:
        lines.append("")
        for f in failures:
            lines.append(f"  {f['case_id']}: {f['reason']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Position card (for /status)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Account overview
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Price board (pinned message)
# ---------------------------------------------------------------------------

def _compact_price(price: float) -> str:
    """Format price compactly: 69123.5 -> '69.1K', 142.35 -> '142'."""
    if price >= 1000:
        return f"{price / 1000:.1f}K"
    return f"{price:,.0f}"


def format_price_board(summaries: list) -> str:
    """Format a price board for pinned message display.

    First line is a compact summary visible in the Telegram pin preview.
    """
    from datetime import UTC, datetime

    if not summaries:
        return "Price Board\nNo symbols configured."

    # Line 1: compact summary for pinned preview
    compact_parts: list[str] = []
    for s in summaries:
        short_symbol = s.symbol.split("/")[0]
        sign = "+" if s.change_24h_pct >= 0 else ""
        compact_parts.append(
            f"{short_symbol}"
            f" {_compact_price(s.price)}"
            f"({sign}{s.change_24h_pct:.1f}%)"
        )
    summary_line = " ".join(compact_parts)

    # Detailed lines
    lines = [summary_line, "", "Price Board"]
    for s in summaries:
        display_symbol = s.symbol.replace(":USDT", "")
        sign = "+" if s.change_24h_pct >= 0 else ""
        lines.append(
            f"{display_symbol}  ${s.price:,.1f}"
            f"  {sign}{s.change_24h_pct:.2f}%"
        )

    now = datetime.now(UTC).astimezone()
    lines.append(f"\nUpdated: {now.strftime('%H:%M:%S')}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Execution plan (two-section format)
# ---------------------------------------------------------------------------

def format_execution_plan(
    *,
    plan: object,
    confidence: float,
    time_horizon: str,
    analysis_summary: dict[str, str] | None = None,
    rationale: str | None = None,
    model_used: str | None = None,
    expires_minutes: int | None = None,
) -> str:
    """Format an ExecutionPlan as a two-section Telegram message.

    Upper section: concrete trade parameters.
    Lower section: agent analysis summaries + rationale.
    """
    side_str = plan.side.upper()  # type: ignore[attr-defined]
    emoji = _SIDE_EMOJI.get(plan.side, "")  # type: ignore[attr-defined]
    symbol = plan.symbol.replace(":USDT", "")  # type: ignore[attr-defined]

    entry_price: float = plan.entry_price  # type: ignore[attr-defined]
    quantity: float = plan.quantity  # type: ignore[attr-defined]
    notional: float = plan.notional_value  # type: ignore[attr-defined]
    margin: float = plan.margin_required  # type: ignore[attr-defined]
    leverage: int = plan.leverage  # type: ignore[attr-defined]
    margin_mode: str = plan.margin_mode  # type: ignore[attr-defined]
    liq_price: float = plan.liquidation_price  # type: ignore[attr-defined]
    max_loss: float = plan.max_loss  # type: ignore[attr-defined]
    max_loss_pct: float = plan.max_loss_pct  # type: ignore[attr-defined]
    rr: float = plan.risk_reward_ratio  # type: ignore[attr-defined]
    tp_profits: list[float] = plan.tp_profits  # type: ignore[attr-defined]
    order_type: str = plan.entry_order.order_type  # type: ignore[attr-defined]

    # --- Title ---
    lines = [
        f"{emoji} {symbol} {side_str} \u00b7 Confidence {confidence:.0%}",
        "",
    ]

    # Entry
    entry_str = f"${entry_price:,.1f} ({order_type})"
    if order_type == "limit" and plan.entry_order.price is not None:  # type: ignore[attr-defined]
        entry_str = f"${plan.entry_order.price:,.1f} (limit)"  # type: ignore[attr-defined]
    lines.append(f"\U0001f3af Entry:     {entry_str}")
    lines.append(f"\U0001f4e6 Quantity:  {quantity:.4f} (${notional:,.0f})")
    lines.append(
        f"\U0001f4b5 Margin:    ${margin:,.0f} \u00b7 {leverage}x {margin_mode}",
    )
    lines.append(f"\U0001f480 Liq:       ${liq_price:,.1f}")

    # SL
    sl_order = plan.sl_order  # type: ignore[attr-defined]
    if sl_order is not None and sl_order.stop_price is not None:
        sl_price = sl_order.stop_price
        sl_pct = (sl_price - entry_price) / entry_price * 100
        lines.append(f"\u26d4 Stop Loss: ${sl_price:,.1f} ({sl_pct:+.1f}%)")

    # TPs
    tp_orders = plan.tp_orders  # type: ignore[attr-defined]
    for i, tp_order in enumerate(tp_orders):
        if tp_order.stop_price is None:
            continue
        tp_pct = (tp_order.stop_price - entry_price) / entry_price * 100
        close_pct_str = f"{tp_order.quantity / quantity * 100:.0f}%"
        profit_str = ""
        if i < len(tp_profits):
            profit_str = f" \u2192 +${tp_profits[i]:,.0f}"
        lines.append(
            f"\u2705 TP{i + 1}:       ${tp_order.stop_price:,.1f}"
            f" ({tp_pct:+.1f}%) \u2192 close {close_pct_str}{profit_str}",
        )

    # Loss / Profit / R:R
    lines.append("")
    lines.append(f"\u26a0\ufe0f Max Loss: ${max_loss:,.0f} ({max_loss_pct:.1f}%)")
    if tp_profits:
        tp_parts = " / ".join(
            f"TP{i + 1} +${p:,.0f}" for i, p in enumerate(tp_profits)
        )
        lines.append(f"\U0001f4b0 Est. Profit: {tp_parts}")
    lines.append(f"\U0001f4ca Risk/Reward: 1:{rr:.1f}")

    # --- Analysis section (optional) ---
    if analysis_summary:
        lines.append("")

        label_map = {
            "technical": "\U0001f4c8 Technical",
            "positioning": "\U0001f4b9 Positioning",
            "catalyst": "\U0001f4c5 Catalyst",
            "correlation": "\U0001f310 Correlation",
        }
        for key in ("technical", "positioning", "catalyst", "correlation"):
            if key in analysis_summary:
                label = label_map.get(key, key.title())
                summary_lines = analysis_summary[key].split("\n")
                lines.append(f"{label}: {summary_lines[0]}")
                for extra in summary_lines[1:]:
                    lines.append(f"   {extra}")

    if rationale:
        lines.append("")
        lines.append("\U0001f4a1 Rationale")
        lines.append(rationale)

    # Footer
    footer_parts: list[str] = []
    if model_used:
        footer_parts.append(f"Model: {model_used.split('/')[-1]}")
    if expires_minutes is not None:
        footer_parts.append(f"Expires in {expires_minutes} min")
    if footer_parts:
        lines.append(f"\n{' \u00b7 '.join(footer_parts)}")

    return "\n".join(lines)


def format_history(trades: list[PaperTradeRecord]) -> str:
    if not trades:
        return "No closed trades yet."

    lines = ["Recent Trades"]
    for t in trades:
        lines.append(
            f"\n  {t.symbol} {t.side.upper()}"
            f"\n  ${t.entry_price:,.1f} \u2192 ${t.exit_price:,.1f}"
            f" | {_pnl_str(t.pnl)}"
        )
    return "\n".join(lines)
