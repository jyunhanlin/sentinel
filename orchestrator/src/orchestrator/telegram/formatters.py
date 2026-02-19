from __future__ import annotations

from typing import TYPE_CHECKING

from orchestrator.exchange.paper_engine import CloseResult
from orchestrator.risk.checker import RiskResult

if TYPE_CHECKING:
    from orchestrator.pipeline.runner import PipelineResult


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
        "/start - Welcome message\n"
        "/status - Account overview & latest proposals\n"
        "/coin <symbol> - Detailed analysis for a symbol (e.g. /coin BTC)\n"
        "/run - Trigger pipeline for all symbols\n"
        "/run <symbol> - Trigger pipeline for specific symbol\n"
        "/run <symbol> sonnet|opus - Trigger with specific model\n"
        "/history - Recent trade records\n"
        "/resume - Un-pause pipeline after risk pause\n"
        "/help - Show this message"
    )


def format_proposal(result: PipelineResult) -> str:
    if result.proposal is None:
        return f"Pipeline {result.status}: {result.rejection_reason or 'No proposal generated'}"

    p = result.proposal
    status_emoji = {"completed": "NEW", "rejected": "REJECTED", "failed": "FAILED"}.get(
        result.status, result.status.upper()
    )

    lines = [
        f"[{status_emoji}] {p.symbol}",
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
        model_label = result.model_used.split("/")[-1]
        lines.append(f"Model: {model_label}")

    if result.status == "rejected":
        lines.append(f"\nREJECTED: {result.rejection_reason}")

    degraded_agents = []
    if result.sentiment_degraded:
        degraded_agents.append("sentiment")
    if result.market_degraded:
        degraded_agents.append("market")
    if result.proposer_degraded:
        degraded_agents.append("proposer")
    if degraded_agents:
        lines.append(f"\nDegraded: {', '.join(degraded_agents)}")

    return "\n".join(lines)


def format_status(results: list[PipelineResult]) -> str:
    if not results:
        return "No pipeline results yet. Use /run to trigger analysis."

    lines = ["Latest pipeline results:\n"]
    for r in results:
        side = r.proposal.side.value.upper() if r.proposal else "N/A"
        conf = f"{r.proposal.confidence:.0%}" if r.proposal else "N/A"
        lines.append(f"  {r.symbol}: {side} (confidence: {conf}) [{r.status}]")

    return "\n".join(lines)


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


def format_status_from_records(records: list) -> str:
    """Format status from DB TradeProposalRecords (fallback when no in-memory results)."""
    import json

    if not records:
        return "No pipeline results yet. Use /run to trigger analysis."

    lines = ["Latest proposals (from DB):\n"]
    for r in records:
        try:
            proposal = json.loads(r.proposal_json)
            symbol = proposal.get("symbol", "?")
            side = proposal.get("side", "?").upper()
            confidence = proposal.get("confidence", 0)
            conf_str = f"{confidence:.0%}" if isinstance(confidence, float) else str(confidence)
            status = r.risk_check_result or "unknown"
            lines.append(f"  {symbol}: {side} (confidence: {conf_str}) [{status}]")
        except (json.JSONDecodeError, AttributeError):
            lines.append(f"  [parse error] proposal_id={r.proposal_id}")

    return "\n".join(lines)


def format_history(trades: list) -> str:
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
