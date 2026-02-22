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
    return dt.strftime("%m/%d %H:%M")


def format_welcome() -> str:
    return (
        "Welcome to Sentinel Orchestrator!\n\n"
        "I analyze crypto markets using multiple AI models and generate "
        "trade proposals with risk management.\n\n"
        "Use /help to see available commands."
    )


def format_help() -> str:
    return (
        "Available commands:\n\n"
        "/start — Welcome message\n"
        "/status — Account overview & latest proposals\n"
        "/coin <symbol> — Detailed analysis for a symbol (e.g. /coin BTC)\n"
        "/run — Trigger pipeline for all symbols\n"
        "/run <symbol> — Trigger pipeline for specific symbol\n"
        "/run <symbol> sonnet|opus — Trigger with specific model\n"
        "/history — Recent trade records\n"
        "/perf — Performance report (PnL, win rate, Sharpe, etc.)\n"
        "/eval — Run LLM evaluation and show results\n"
        "/resume — Un-pause pipeline after risk pause\n"
        "/help — Show this message"
    )


def format_proposal(result: PipelineResult) -> str:
    if result.proposal is None:
        return f"Pipeline {result.status}: {result.rejection_reason or 'No proposal generated'}"

    p = result.proposal
    is_flat = p.side.value == "flat"
    status_label = {
        "completed": "FLAT" if is_flat else "EXECUTED",
        "rejected": "REJECTED",
        "failed": "FAILED",
        "pending_approval": "PENDING APPROVAL",
    }.get(result.status, result.status.upper())

    time_str = _fmt_time(result.created_at)
    lines = [
        f"[{status_label}] {p.symbol}  ({time_str})",
        f"Side: {p.side.value.upper()}",
    ]

    if p.side.value != "flat":
        lines.append(f"Entry: {p.entry.type}")
        lines.append(f"Risk: {p.position_size_risk_pct}%")
        if p.stop_loss is not None:
            lines.append(f"SL: {p.stop_loss:,.1f}")
        if p.take_profit:
            tp_str = ", ".join(f"{tp:,.1f}" for tp in p.take_profit)
            lines.append(f"TP: {tp_str}")

    lines.append(f"Horizon: {p.time_horizon}")
    lines.append(f"Confidence: {p.confidence:.0%}")
    lines.append(f"Rationale: {p.rationale}")

    if result.model_used:
        lines.append(f"Model: {result.model_used.split('/')[-1]}")

    if result.status == "rejected":
        lines.append(f"\nREJECTED: {result.rejection_reason}")

    degraded = _degraded_labels(result)
    if degraded:
        lines.append(f"\nDegraded: {', '.join(degraded)}")

    return "\n".join(lines)


def format_status(results: list[PipelineResult]) -> str:
    if not results:
        return "No pipeline results yet. Use /run to trigger analysis."

    blocks = ["Latest pipeline results:\n─────────────────────"]
    for r in results:
        side = r.proposal.side.value.upper() if r.proposal else "N/A"
        time_str = _fmt_time(r.created_at)
        blocks.append(f"{r.symbol} — {side} [{r.status}]  ({time_str})")

        if r.proposal is None:
            reason = r.rejection_reason or "No proposal generated"
            blocks.append(f"  {reason}")
        elif r.proposal.side.value == "flat":
            blocks.append(f"  Confidence: {r.proposal.confidence:.0%}")
            blocks.append(f"  Rationale: {_truncate(r.proposal.rationale, 80)}")
        else:
            p = r.proposal
            blocks.append(f"  Entry: {p.entry.type} | Risk: {p.position_size_risk_pct}%")
            sl_str = f"${p.stop_loss:,.1f}" if p.stop_loss is not None else "—"
            tp_str = ", ".join(f"${tp:,.1f}" for tp in p.take_profit) if p.take_profit else "—"
            blocks.append(f"  SL: {sl_str} | TP: {tp_str}")
            blocks.append(f"  Confidence: {p.confidence:.0%} | Horizon: {p.time_horizon}")

        if r.model_used:
            blocks.append(f"  Model: {r.model_used.split('/')[-1]}")

        degraded = _degraded_labels(r)
        if degraded:
            blocks.append(f"  Degraded: {', '.join(degraded)}")

        blocks.append("")  # blank line separator

    return "\n".join(blocks).rstrip()


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


def format_trade_report(result: CloseResult) -> str:
    reason_label = {"sl": "SL", "tp": "TP"}.get(result.reason, result.reason.upper())
    pnl_str = f"${result.pnl:,.2f}" if result.pnl >= 0 else f"-${abs(result.pnl):,.2f}"

    lines = [
        f"[CLOSED] {result.symbol}",
        f"Side: {result.side.value.upper()}",
        f"Entry: {result.entry_price:,.1f} → Exit: {result.exit_price:,.1f} ({reason_label})",
        f"Quantity: {result.quantity:.4f}",
        f"PnL: {pnl_str} (fees: ${result.fees:,.2f})",
    ]
    return "\n".join(lines)


def format_risk_rejection(
    *, symbol: str, side: str, entry_price: float, risk_result: RiskResult
) -> str:
    label = "RISK PAUSED" if risk_result.action == "pause" else "RISK REJECTED"
    lines = [
        f"[{label}] {symbol}",
        f"Proposed: {side} @ {entry_price:,.1f}",
        f"Rule: {risk_result.rule_violated}",
        f"Reason: {risk_result.reason}",
    ]
    return "\n".join(lines)


def format_status_from_records(records: list[TradeProposalRecord]) -> str:
    """Format status from DB TradeProposalRecords (fallback when no in-memory results)."""
    import json

    if not records:
        return "No pipeline results yet. Use /run to trigger analysis."

    blocks = ["Latest proposals (from DB):\n─────────────────────"]
    for r in records:
        try:
            p = json.loads(r.proposal_json)
            symbol = p.get("symbol", "?")
            side = p.get("side", "?").upper()
            status = r.risk_check_result or "unknown"
            time_str = _fmt_time(r.created_at)
            blocks.append(f"{symbol} — {side} [{status}]  ({time_str})")

            confidence = p.get("confidence", 0)
            conf_str = f"{confidence:.0%}" if isinstance(confidence, float) else str(confidence)

            if side == "FLAT":
                blocks.append(f"  Confidence: {conf_str}")
                rationale = p.get("rationale", "")
                if rationale:
                    blocks.append(f"  Rationale: {_truncate(rationale, 80)}")
            else:
                entry_type = p.get("entry", {}).get("type", "?")
                risk_pct = p.get("position_size_risk_pct", "?")
                blocks.append(f"  Entry: {entry_type} | Risk: {risk_pct}%")

                sl = p.get("stop_loss")
                sl_str = f"${sl:,.1f}" if sl is not None else "—"
                tps = p.get("take_profit", [])
                tp_str = ", ".join(f"${tp:,.1f}" for tp in tps) if tps else "—"
                blocks.append(f"  SL: {sl_str} | TP: {tp_str}")

                horizon = p.get("time_horizon", "?")
                blocks.append(f"  Confidence: {conf_str} | Horizon: {horizon}")

            blocks.append("")  # blank line separator
        except (json.JSONDecodeError, AttributeError):
            blocks.append(f"[parse error] proposal_id={r.proposal_id}\n")

    return "\n".join(blocks).rstrip()


def format_perf_report(stats: PerformanceStats) -> str:
    if stats.total_trades == 0:
        return "No trades yet. Performance report will be available after closing positions."

    pnl_sign = "+" if stats.total_pnl >= 0 else "-"
    pnl_str = f"{pnl_sign}${abs(stats.total_pnl):,.2f}"
    pnl_pct_sign = "+" if stats.total_pnl_pct >= 0 else ""
    pf_str = "inf" if stats.profit_factor == float("inf") else f"{stats.profit_factor:.2f}"

    lines = [
        "Performance Report",
        "─────────────────────",
        f"Total PnL:      {pnl_str} ({pnl_pct_sign}{stats.total_pnl_pct:.1f}%)",
        f"Win Rate:       {stats.win_rate:.1%} ({stats.winning_trades}/{stats.total_trades})",
        f"Profit Factor:  {pf_str}",
        f"Max Drawdown:   {stats.max_drawdown_pct:.1f}%",
        f"Sharpe Ratio:   {stats.sharpe_ratio:.2f}",
        "─────────────────────",
    ]
    return "\n".join(lines)


def format_eval_report(report: dict[str, Any]) -> str:
    lines = [
        f"Eval Report ({report['dataset_name']})",
        "──────────────────────────",
        f"Cases: {report['total_cases']} | "
        f"Passed: {report['passed_cases']} | Failed: {report['failed_cases']}",
        f"Accuracy: {report['accuracy']:.0%}",
    ]
    if report.get("consistency_score") is not None:
        lines.append(f"Consistency: {report['consistency_score']:.1%}")
    lines.append("──────────────────────────")

    for f in report.get("failures", []):
        lines.append(f"  {f['case_id']}: {f['reason']}")

    return "\n".join(lines)


def format_pending_approval(approval: PendingApproval) -> str:
    """Format a PendingApproval for TG push with inline keyboard context."""
    p = approval.proposal
    lines = [
        f"[PENDING APPROVAL] {p.symbol}",
        f"Side: {p.side.value.upper()}",
        f"Entry: {p.entry.type} @ ~${approval.snapshot_price:,.1f}",
        f"Risk: {p.position_size_risk_pct}%",
    ]
    if p.stop_loss is not None:
        lines.append(f"SL: ${p.stop_loss:,.1f}")
    if p.take_profit:
        tp_str = ", ".join(f"${tp:,.1f}" for tp in p.take_profit)
        lines.append(f"TP: {tp_str}")
    lines.append(f"Confidence: {p.confidence:.0%}")
    lines.append(f"Rationale: {p.rationale}")

    if approval.model_used:
        lines.append(f"Model: {approval.model_used.split('/')[-1]}")

    remaining = int((approval.expires_at - approval.created_at).total_seconds() / 60)
    lines.append(f"\nExpires in {remaining} minutes")
    return "\n".join(lines)


def format_execution_result(result: ExecutionResult) -> str:
    """Format an ExecutionResult for TG confirmation."""
    lines = [
        f"[EXECUTED] {result.symbol} {result.side.upper()}",
        f"Mode: {result.mode}",
        f"Entry: ${result.entry_price:,.1f} | Qty: {result.quantity:.4f}",
        f"Fees: ${result.fees:,.2f}",
    ]
    if result.sl_order_id:
        lines.append(f"SL order: {result.sl_order_id}")
    if result.tp_order_id:
        lines.append(f"TP order: {result.tp_order_id}")
    return "\n".join(lines)


def format_history(trades: list[PaperTradeRecord]) -> str:
    if not trades:
        return "No closed trades yet."

    lines = ["Recent closed trades:\n"]
    for t in trades:
        pnl_str = f"${t.pnl:,.2f}" if t.pnl >= 0 else f"-${abs(t.pnl):,.2f}"
        lines.append(
            f"  {t.symbol} {t.side.upper()} | "
            f"{t.entry_price:,.1f} → {t.exit_price:,.1f} | "
            f"PnL: {pnl_str}"
        )
    return "\n".join(lines)
