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


def format_welcome() -> str:
    return (
        "Welcome to Sentinel!\n\n"
        "I analyze crypto markets using multiple AI models and generate "
        "trade proposals with risk management.\n\n"
        "Use /help to see available commands."
    )


def format_help() -> str:
    return (
        "━━ Available Commands ━━\n\n"
        "/start — Welcome message\n"
        "/status — Account overview & latest proposals\n"
        "/coin <symbol> — Detailed analysis (e.g. /coin BTC)\n"
        "/run — Trigger pipeline for all symbols\n"
        "/run <symbol> — Trigger for specific symbol\n"
        "/run <symbol> sonnet|opus — Trigger with specific model\n"
        "/history — Recent trade records\n"
        "/perf — Performance report\n"
        "/eval — Run LLM evaluation\n"
        "/resume — Un-pause after risk pause\n"
        "/help — Show this message"
    )


def _format_tp_lines(take_profit: list, entry_price: float | None = None) -> list[str]:
    """Format take-profit levels with optional % from entry."""
    lines: list[str] = []
    for i, tp in enumerate(take_profit, 1):
        price = tp.price if hasattr(tp, "price") else tp
        close_pct = tp.close_pct if hasattr(tp, "close_pct") else 100
        pct_str = ""
        if entry_price and entry_price > 0:
            pct = (price - entry_price) / entry_price * 100
            pct_str = f" ({pct:+.1f}%)"
        lines.append(f"TP{i}:   ${price:,.1f}{pct_str} → close {close_pct}%")
    return lines


def format_proposal(result: PipelineResult) -> str:
    if result.proposal is None:
        return (
            f"━━ {result.status.upper()} ━━\n"
            f"{result.rejection_reason or 'No proposal generated'}"
        )

    p = result.proposal
    is_flat = p.side.value == "flat"
    status_label = {
        "completed": "FLAT" if is_flat else "EXECUTED",
        "rejected": "REJECTED",
        "failed": "FAILED",
        "pending_approval": "PENDING APPROVAL",
    }.get(result.status, result.status.upper())

    time_str = _fmt_time(result.created_at)
    side_emoji = {"long": "\U0001f7e2", "short": "\U0001f534", "flat": "\u26aa"}.get(
        p.side.value, ""
    )

    lines = [f"━━ {status_label} ━━"]

    if is_flat:
        lines.append(f"{side_emoji} {p.symbol}")
    else:
        lines.append(f"{side_emoji} {p.side.value.upper()} {p.symbol}")

    lines.append(time_str)

    if not is_flat:
        entry_price = p.entry.price if p.entry.price else None
        lines.append("")
        entry_type_str = p.entry.type
        if p.entry.price:
            entry_type_str += f" @ ${p.entry.price:,.1f}"
        lines.append(f"Entry: {entry_type_str}")
        if p.stop_loss is not None:
            sl_pct = ""
            if entry_price and entry_price > 0:
                sl_pct = f" ({(p.stop_loss - entry_price) / entry_price * 100:+.1f}%)"
            lines.append(f"SL:    ${p.stop_loss:,.1f}{sl_pct}")
        for tp_line in _format_tp_lines(p.take_profit, entry_price):
            lines.append(tp_line)
        lines.append(f"Leverage: {p.suggested_leverage}x")

    lines.append("")
    lines.append(f"Confidence: {p.confidence:.0%}")
    lines.append(f"Horizon: {p.time_horizon}")
    lines.append(f"\nRationale: {p.rationale}")

    if result.model_used:
        lines.append(f"\nModel: {result.model_used.split('/')[-1]}")

    if result.status == "rejected":
        lines.append(f"\nREJECTED: {result.rejection_reason}")

    degraded = _degraded_labels(result)
    if degraded:
        lines.append(f"Degraded: {', '.join(degraded)}")

    return "\n".join(lines)


def format_pending_approval(
    approval: PendingApproval,
    *,
    sentiment: Any | None = None,
    market: Any | None = None,
) -> str:
    """Format a PendingApproval for TG push with detailed analysis report."""
    p = approval.proposal
    entry_price = approval.snapshot_price

    side_emoji = {"long": "\U0001f7e2", "short": "\U0001f534"}.get(p.side.value, "")
    lines = [
        f"{side_emoji} {p.side.value.upper()} {p.symbol}",
        "\u2500" * 22,
        f"\u25b6 Entry:  ${entry_price:,.1f} ({p.entry.type})",
    ]

    if p.stop_loss is not None:
        sl_pct = (p.stop_loss - entry_price) / entry_price * 100
        lines.append(f"\u26d4 SL:     ${p.stop_loss:,.1f} ({sl_pct:+.1f}%)")

    for i, tp in enumerate(p.take_profit, 1):
        tp_pct = (tp.price - entry_price) / entry_price * 100
        lines.append(f"\u2705 TP{i}:    ${tp.price:,.1f} ({tp_pct:+.1f}%) \u2192 close {tp.close_pct}%")

    lines.append("\u2500" * 22)

    # Leverage and R:R
    vol_str = ""
    if market and hasattr(market, "volatility_pct"):
        vol_str = f" (vol: {market.volatility_pct:.1f}%)"
    lines.append(f"Leverage: {p.suggested_leverage}x{vol_str}")

    if p.stop_loss is not None and p.take_profit:
        risk_dist = abs(entry_price - p.stop_loss)
        reward_dist = abs(p.take_profit[-1].price - entry_price)
        rr = reward_dist / risk_dist if risk_dist > 0 else 0
        lines.append(f"Risk/Reward: 1:{rr:.1f}")

    lines.append(f"Confidence: {p.confidence:.0%}")
    lines.append("\u2500" * 22)

    # Market analysis section
    if market:
        lines.append("")
        trend_str = market.trend.value.upper() if hasattr(market.trend, "value") else str(market.trend).upper()
        vol_regime = market.volatility_regime.value.upper() if hasattr(market.volatility_regime, "value") else str(market.volatility_regime).upper()
        vol_pct = f" ({market.volatility_pct:.1f}%)" if hasattr(market, "volatility_pct") else ""
        lines.append(f"\U0001f4ca Market:")
        lines.append(f"Trend: {trend_str} | Vol: {vol_regime}{vol_pct}")

        supports = [kl for kl in market.key_levels if kl.type == "support"]
        resists = [kl for kl in market.key_levels if kl.type == "resistance"]
        if supports:
            lines.append(f"Support: {' / '.join(f'{kl.price:,.0f}' for kl in supports)}")
        if resists:
            lines.append(f"Resist:  {' / '.join(f'{kl.price:,.0f}' for kl in resists)}")

        if market.risk_flags:
            lines.append(f"\n\u26a0\ufe0f Risk: {', '.join(market.risk_flags)}")
        else:
            lines.append(f"\n\u26a0\ufe0f Risk: No flags")

    # Sentiment section
    if sentiment:
        score = sentiment.sentiment_score
        label = "bullish" if score > 60 else "bearish" if score < 40 else "neutral"
        lines.append(f"\n\U0001f5e3 Sentiment: {score}/100 ({label})")
        if sentiment.key_events:
            lines.append(f"Key: {sentiment.key_events[0].event}")

    lines.append(f"\nRationale: {p.rationale}")

    if approval.model_used:
        lines.append(f"\nModel: {approval.model_used.split('/')[-1]}")

    remaining = int((approval.expires_at - approval.created_at).total_seconds() / 60)
    lines.append(f"Expires in {remaining} minutes")

    return "\n".join(lines)


def format_execution_result(result: ExecutionResult) -> str:
    """Format an ExecutionResult for TG confirmation."""
    lines = [
        "━━ EXECUTED ━━",
        f"{result.symbol}  {result.side.upper()}",
        f"Mode: {result.mode}",
        "",
        f"Entry: ${result.entry_price:,.1f} | Qty: {result.quantity:.4f}",
        f"Fees:  ${result.fees:,.2f}",
    ]
    if result.sl_order_id:
        lines.append(f"SL order: {result.sl_order_id}")
    if result.tp_order_id:
        lines.append(f"TP order: {result.tp_order_id}")
    return "\n".join(lines)


def format_trade_report(result: CloseResult) -> str:
    reason_label = {
        "sl": "SL", "tp": "TP", "liquidation": "LIQUIDATION",
        "manual": "MANUAL", "partial_reduce": "PARTIAL",
    }.get(result.reason, result.reason.upper())
    if result.partial:
        reason_label = f"PARTIAL {reason_label}"
    pnl_sign = "+" if result.pnl >= 0 else "-"
    pnl_str = f"{pnl_sign}${abs(result.pnl):,.2f}"

    lines = [
        f"━━ CLOSED ({reason_label}) ━━",
        f"{result.symbol}  {result.side.value.upper()}",
        f"${result.entry_price:,.1f} \u2192 ${result.exit_price:,.1f}",
        "",
        f"Qty: {result.quantity:.4f}",
        f"PnL: {pnl_str} (fees: ${result.fees:,.2f})",
    ]
    if result.remaining_quantity is not None:
        lines.append(f"Remaining: {result.remaining_quantity:.4f}")
    return "\n".join(lines)


def format_risk_rejection(
    *, symbol: str, side: str, entry_price: float, risk_result: RiskResult
) -> str:
    label = "RISK PAUSED" if risk_result.action == "pause" else "RISK REJECTED"
    lines = [
        f"━━ {label} ━━",
        symbol,
        f"Proposed: {side} @ ${entry_price:,.1f}",
        "",
        f"Rule: {risk_result.rule_violated}",
        f"Reason: {risk_result.reason}",
    ]
    return "\n".join(lines)


def format_status(results: list[PipelineResult]) -> str:
    if not results:
        return "No pipeline results yet. Use /run to trigger analysis."

    blocks = ["━━ Pipeline Status ━━"]
    for r in results:
        side = r.proposal.side.value.upper() if r.proposal else "N/A"
        time_str = _fmt_time(r.created_at)
        blocks.append(f"\n{r.symbol} — {side} [{r.status}]  ({time_str})")

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
                f"  Entry: {p.entry.type} | Risk: {p.position_size_risk_pct}%"
            )
            sl_str = (
                f"${p.stop_loss:,.1f}" if p.stop_loss is not None else "—"
            )
            tp_str = (
                ", ".join(f"${tp.price:,.1f}" for tp in p.take_profit)
                if p.take_profit
                else "—"
            )
            blocks.append(f"  SL: {sl_str} | TP: {tp_str}")
            blocks.append(
                f"  Confidence: {p.confidence:.0%} | Horizon: {p.time_horizon}"
            )

        if r.model_used:
            blocks.append(f"  Model: {r.model_used.split('/')[-1]}")

        degraded = _degraded_labels(r)
        if degraded:
            blocks.append(f"  Degraded: {', '.join(degraded)}")

    return "\n".join(blocks)


def format_status_from_records(records: list[TradeProposalRecord]) -> str:
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
            blocks.append(f"\n{symbol} — {side} [{status}]  ({time_str})")

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
                        f"  Rationale: {_truncate(rationale, 80)}"
                    )
            else:
                entry_type = p.get("entry", {}).get("type", "?")
                risk_pct = p.get("position_size_risk_pct", "?")
                blocks.append(f"  Entry: {entry_type} | Risk: {risk_pct}%")

                sl = p.get("stop_loss")
                sl_str = f"${sl:,.1f}" if sl is not None else "—"
                tps = p.get("take_profit", [])
                tp_parts: list[str] = []
                for tp in tps:
                    if isinstance(tp, dict):
                        tp_parts.append(f"${tp['price']:,.1f}")
                    else:
                        tp_parts.append(f"${tp:,.1f}")
                tp_str = ", ".join(tp_parts) if tp_parts else "\u2014"
                blocks.append(f"  SL: {sl_str} | TP: {tp_str}")

                horizon = p.get("time_horizon", "?")
                blocks.append(
                    f"  Confidence: {conf_str} | Horizon: {horizon}"
                )

        except (json.JSONDecodeError, AttributeError):
            blocks.append(f"\n[parse error] proposal_id={r.proposal_id}")

    return "\n".join(blocks)


def format_perf_report(stats: PerformanceStats) -> str:
    if stats.total_trades == 0:
        return (
            "No trades yet. Performance report will be "
            "available after closing positions."
        )

    pnl_sign = "+" if stats.total_pnl >= 0 else "-"
    pnl_str = f"{pnl_sign}${abs(stats.total_pnl):,.2f}"
    pnl_pct_sign = "+" if stats.total_pnl_pct >= 0 else ""
    pf_str = (
        "inf"
        if stats.profit_factor == float("inf")
        else f"{stats.profit_factor:.2f}"
    )

    lines = [
        "━━ Performance Report ━━",
        "",
        f"Total PnL:     {pnl_str} ({pnl_pct_sign}{stats.total_pnl_pct:.1f}%)",
        f"Win Rate:      {stats.win_rate:.1%} ({stats.winning_trades}/{stats.total_trades})",
        f"Profit Factor: {pf_str}",
        f"Max Drawdown:  {stats.max_drawdown_pct:.1f}%",
        f"Sharpe Ratio:  {stats.sharpe_ratio:.2f}",
    ]
    return "\n".join(lines)


def format_eval_report(report: dict[str, Any]) -> str:
    lines = [
        f"━━ Eval Report ({report['dataset_name']}) ━━",
        "",
        f"Cases: {report['total_cases']} | "
        f"Passed: {report['passed_cases']} | "
        f"Failed: {report['failed_cases']}",
        f"Accuracy: {report['accuracy']:.0%}",
    ]
    if report.get("consistency_score") is not None:
        lines.append(f"Consistency: {report['consistency_score']:.1%}")

    failures = report.get("failures", [])
    if failures:
        lines.append("")
        for f in failures:
            lines.append(f"  {f['case_id']}: {f['reason']}")

    return "\n".join(lines)


def format_position_card(info: dict) -> str:
    """Format a single position with PnL info for /status display."""
    pos = info["position"]
    pnl = info["unrealized_pnl"]
    pnl_pct = info["pnl_pct"]
    roe_pct = info["roe_pct"]

    pnl_sign = "+" if pnl >= 0 else ""
    side_str = pos.side.value.upper() if hasattr(pos.side, "value") else str(pos.side).upper()
    leverage_str = f" {pos.leverage}x" if pos.leverage > 1 else ""

    lines = [
        f"{pos.symbol}  {side_str}{leverage_str}",
        f"Entry: ${pos.entry_price:,.1f} | Qty: {pos.quantity:.4f}",
    ]
    if pos.margin > 0:
        lines.append(f"Margin: ${pos.margin:,.2f} | Liq: ${pos.liquidation_price:,.1f}")
    lines.append(f"SL: ${pos.stop_loss:,.1f}")
    if pos.take_profit:
        tp_str = ", ".join(f"${tp.price:,.1f} ({tp.close_pct}%)" for tp in pos.take_profit)
        lines.append(f"TP: {tp_str}")

    lines.append(f"PnL: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)")
    if pos.margin > 0:
        lines.append(f"ROE: {pnl_sign}{roe_pct:.2f}%")

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
        "━━ Account Overview ━━",
        "",
        f"Equity:      ${equity:,.2f} ({pnl_sign}${total_pnl:,.2f})",
        f"Available:   ${available:,.2f}",
        f"Used Margin: ${used_margin:,.2f}",
    ]

    if position_count > 0:
        lines.append(f"\nOpen Positions: {position_count}")

    return "\n".join(lines)


def format_history_paginated(
    trades: list[PaperTradeRecord],
    page: int,
    total_pages: int,
) -> str:
    """Format closed trades with pagination info."""
    if not trades:
        return "No closed trades yet."

    lines = [f"━━ Trade History — Page {page}/{total_pages} ━━"]
    for t in trades:
        pnl_sign = "+" if t.pnl >= 0 else "-"
        pnl_str = f"{pnl_sign}${abs(t.pnl):,.2f}"
        side_str = t.side.upper() if isinstance(t.side, str) else t.side.value.upper()
        leverage_str = f" {t.leverage}x" if t.leverage > 1 else ""
        reason_str = f" ({t.close_reason.upper()})" if t.close_reason else ""

        lines.append(f"\n  {t.symbol} {side_str}{leverage_str}{reason_str}")
        lines.append(f"  ${t.entry_price:,.1f} → ${t.exit_price:,.1f} | PnL: {pnl_str}")

        if t.margin > 0:
            roe = (t.pnl / t.margin * 100) if t.margin else 0
            roe_sign = "+" if roe >= 0 else ""
            lines.append(f"  Margin: ${t.margin:,.2f} | ROE: {roe_sign}{roe:.2f}%")

    return "\n".join(lines)


def _compact_price(price: float) -> str:
    """Format price compactly: 69123.5 → '69.1K', 2530.8 → '2.5K', 142.35 → '142'."""
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
        short_symbol = s.symbol.split("/")[0]  # "BTC/USDT:USDT" → "BTC"
        sign = "+" if s.change_24h_pct >= 0 else ""
        compact_parts.append(
            f"{short_symbol} {_compact_price(s.price)}({sign}{s.change_24h_pct:.1f}%)"
        )
    summary_line = " ".join(compact_parts)

    # Detailed lines
    lines = [summary_line, "━━ Price Board ━━"]
    for s in summaries:
        display_symbol = s.symbol.replace(":USDT", "")
        sign = "+" if s.change_24h_pct >= 0 else ""
        lines.append(f"{display_symbol}  ${s.price:,.1f}  {sign}{s.change_24h_pct:.2f}%")

    now = datetime.now(UTC).astimezone()
    lines.append(f"\nUpdated: {now.strftime('%H:%M:%S')}")

    return "\n".join(lines)


def format_history(trades: list[PaperTradeRecord]) -> str:
    if not trades:
        return "No closed trades yet."

    lines = ["━━ Recent Trades ━━"]
    for t in trades:
        pnl_sign = "+" if t.pnl >= 0 else "-"
        pnl_str = f"{pnl_sign}${abs(t.pnl):,.2f}"
        lines.append(
            f"\n  {t.symbol} {t.side.upper()}"
            f"\n  ${t.entry_price:,.1f} → ${t.exit_price:,.1f} | PnL: {pnl_str}"
        )
    return "\n".join(lines)
