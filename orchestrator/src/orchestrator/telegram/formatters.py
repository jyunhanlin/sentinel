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
    if result.sentiment_degraded:
        labels.append("sentiment")
    if result.market_degraded:
        labels.append("market")
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
        "━━ Commands ━━\n\n"
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
            f"\u2705 TP{i}: ${price:,.1f}{pct_str} \u2192 {close_pct}%"
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
            f"Conf {p.confidence:.0%} | {p.time_horizon}",
            f"\n{p.rationale}",
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
            f"\u25b6 Entry: {p.entry.type}"
            + (f" @ ${p.entry.price:,.1f}" if p.entry.price else ""),
        ]
        if p.stop_loss is not None:
            sl_pct = ""
            if entry_price and entry_price > 0:
                pct = (p.stop_loss - entry_price) / entry_price * 100
                sl_pct = f" ({pct:+.1f}%)"
            lines.append(f"\u26d4 SL: ${p.stop_loss:,.1f}{sl_pct}")
        lines.extend(_format_tp_lines(p.take_profit, entry_price))
        lines.append(
            f"\n{p.suggested_leverage}x"
            f" \u00b7 Conf {p.confidence:.0%}"
            f" \u00b7 {p.time_horizon}"
        )
        lines.append(f"\n{p.rationale}")

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
    sentiment: Any | None = None,
    market: Any | None = None,
) -> str:
    """Format a PendingApproval with detailed analysis report."""
    p = approval.proposal
    entry_price = approval.snapshot_price
    emoji = _SIDE_EMOJI.get(p.side.value, "")

    lines = [
        f"{emoji} {p.side.value.upper()} {p.symbol}",
        "",
        f"\u25b6 Entry:  ${entry_price:,.1f} ({p.entry.type})",
    ]

    if p.stop_loss is not None:
        sl_pct = (p.stop_loss - entry_price) / entry_price * 100
        lines.append(
            f"\u26d4 SL:     ${p.stop_loss:,.1f} ({sl_pct:+.1f}%)"
        )

    for i, tp in enumerate(p.take_profit, 1):
        tp_pct = (tp.price - entry_price) / entry_price * 100
        lines.append(
            f"\u2705 TP{i}: ${tp.price:,.1f}"
            f" ({tp_pct:+.1f}%) \u2192 {tp.close_pct}%"
        )

    # Summary line: leverage · R:R · confidence
    parts = [f"{p.suggested_leverage}x"]
    if market and hasattr(market, "volatility_pct"):
        parts[0] += f" (vol {market.volatility_pct:.1f}%)"
    if p.stop_loss is not None and p.take_profit:
        risk_dist = abs(entry_price - p.stop_loss)
        reward_dist = abs(p.take_profit[-1].price - entry_price)
        rr = reward_dist / risk_dist if risk_dist > 0 else 0
        parts.append(f"R:R 1:{rr:.1f}")
    parts.append(f"Conf {p.confidence:.0%}")
    lines.append(f"\n{' \u00b7 '.join(parts)}")

    # Market section (compact)
    if market:
        trend_str = (
            market.trend.value.upper()
            if hasattr(market.trend, "value")
            else str(market.trend).upper()
        )
        vol_regime = (
            market.volatility_regime.value.upper()
            if hasattr(market.volatility_regime, "value")
            else str(market.volatility_regime).upper()
        )
        lines.append(f"\n\U0001f4ca {trend_str} | {vol_regime}")

        supports = [
            kl for kl in market.key_levels if kl.type == "support"
        ]
        resists = [
            kl for kl in market.key_levels if kl.type == "resistance"
        ]
        level_parts: list[str] = []
        if supports:
            s_str = " / ".join(f"{kl.price:,.0f}" for kl in supports)
            level_parts.append(f"S: {s_str}")
        if resists:
            r_str = " / ".join(f"{kl.price:,.0f}" for kl in resists)
            level_parts.append(f"R: {r_str}")
        if level_parts:
            lines.append(" | ".join(level_parts))

        if market.risk_flags:
            flags = ", ".join(market.risk_flags)
            lines.append(f"\u26a0\ufe0f {flags}")

    # Sentiment section (compact)
    if sentiment:
        score = sentiment.sentiment_score
        label = (
            "bullish" if score > 60
            else "bearish" if score < 40
            else "neutral"
        )
        line = f"\n\U0001f5e3 {score}/100 {label}"
        if sentiment.key_events:
            line += f" — {sentiment.key_events[0].event}"
        lines.append(line)

    lines.append(f"\n{p.rationale}")

    # Footer
    footer_parts: list[str] = []
    if approval.model_used:
        footer_parts.append(approval.model_used.split("/")[-1])
    remaining = int(
        (approval.expires_at - approval.created_at).total_seconds() / 60
    )
    footer_parts.append(f"{remaining}m")
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
        f"${result.entry_price:,.1f} | {result.quantity:.4f}",
        f"Fees: ${result.fees:,.2f}",
    ]
    if result.sl_order_id:
        lines.append(f"SL order: {result.sl_order_id}")
    if result.tp_order_id:
        lines.append(f"TP order: {result.tp_order_id}")
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
        f"{pnl_emoji} {reason_label} — {result.symbol} {side_str}",
        f"${result.entry_price:,.1f} \u2192 ${result.exit_price:,.1f}",
        f"PnL: {_pnl_str(result.pnl)} (fees ${result.fees:,.2f})",
    ]
    if result.remaining_quantity is not None:
        lines.append(f"Remaining: {result.remaining_quantity:.4f}")
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


# ---------------------------------------------------------------------------
# Status / overview
# ---------------------------------------------------------------------------

def format_status(results: list[PipelineResult]) -> str:
    if not results:
        return "No pipeline results yet. Use /run to trigger analysis."

    blocks = ["━━ Pipeline Status ━━"]
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
            blocks.append(f"  SL: {sl_str} | TP: {tp_str}")
            blocks.append(
                f"  Conf {p.confidence:.0%} | {p.time_horizon}"
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

    blocks = ["━━ Pipeline Status (DB) ━━"]
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
                blocks.append(f"  SL: {sl_str} | TP: {tp_str}")
                horizon = p.get("time_horizon", "?")
                blocks.append(f"  Conf {conf_str} | {horizon}")

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
        "━━ Performance ━━",
        "",
        f"PnL:         {pnl_str}"
        f" ({pnl_pct_sign}{stats.total_pnl_pct:.1f}%)",
        f"Win Rate:    {stats.win_rate:.1%}"
        f" ({stats.winning_trades}/{stats.total_trades})",
        f"Profit Factor: {pf_str}",
        f"Max DD:      {stats.max_drawdown_pct:.1f}%",
        f"Sharpe:      {stats.sharpe_ratio:.2f}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Eval report
# ---------------------------------------------------------------------------

def format_eval_report(report: dict[str, Any]) -> str:
    lines = [
        f"━━ Eval: {report['dataset_name']} ━━",
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
        f"Entry: ${pos.entry_price:,.1f} | Qty: {pos.quantity:.4f}",
    ]
    if pos.margin > 0:
        lines.append(
            f"Margin: ${pos.margin:,.2f}"
            f" | Liq: ${pos.liquidation_price:,.1f}"
        )
    lines.append(f"SL: ${pos.stop_loss:,.1f}")
    if pos.take_profit:
        tp_str = ", ".join(
            f"${tp.price:,.1f} ({tp.close_pct}%)"
            for tp in pos.take_profit
        )
        lines.append(f"TP: {tp_str}")

    lines.append(
        f"PnL: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)"
    )
    if pos.margin > 0:
        lines.append(f"ROE: {pnl_sign}{roe_pct:.2f}%")

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
        "━━ Account Overview ━━",
        "",
        f"Equity:      ${equity:,.2f}"
        f" ({pnl_sign}${total_pnl:,.2f})",
        f"Available:   ${available:,.2f}",
        f"Used Margin: ${used_margin:,.2f}",
    ]

    if position_count > 0:
        lines.append(f"\nOpen Positions: {position_count}")

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

    lines = [f"━━ History — {page}/{total_pages} ━━"]
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
            f"  ${t.entry_price:,.1f} \u2192 ${t.exit_price:,.1f}"
            f" | {_pnl_str(t.pnl)}"
        )

        if t.margin > 0:
            roe = (t.pnl / t.margin * 100) if t.margin else 0
            roe_sign = "+" if roe >= 0 else ""
            lines.append(
                f"  Margin: ${t.margin:,.2f}"
                f" | ROE: {roe_sign}{roe:.2f}%"
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
        return "━━ Price Board ━━\nNo symbols configured."

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
    lines = [summary_line, "", "━━ Price Board ━━"]
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


def format_history(trades: list[PaperTradeRecord]) -> str:
    if not trades:
        return "No closed trades yet."

    lines = ["━━ Recent Trades ━━"]
    for t in trades:
        lines.append(
            f"\n  {t.symbol} {t.side.upper()}"
            f"\n  ${t.entry_price:,.1f} \u2192 ${t.exit_price:,.1f}"
            f" | {_pnl_str(t.pnl)}"
        )
    return "\n".join(lines)
