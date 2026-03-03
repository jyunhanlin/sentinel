from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime

from pydantic import BaseModel

from orchestrator.storage.repository import PaperTradeRepository, TradeProposalRepository


class TradeEvaluation(BaseModel, frozen=True):
    """Single trade vs proposal comparison."""

    trade_id: str
    proposal_id: str
    symbol: str
    direction_correct: bool
    entry_deviation_pct: float | None
    close_reason: str
    confidence: int
    pnl: float


class SymbolStats(BaseModel, frozen=True):
    symbol: str
    total_trades: int
    direction_accuracy: float
    avg_entry_deviation_pct: float
    sl_hit_rate: float
    tp_hit_rate: float
    total_pnl: float


class ConfidenceBucket(BaseModel, frozen=True):
    bucket: str
    total_trades: int
    direction_accuracy: float
    avg_pnl: float


class PeriodStats(BaseModel, frozen=True):
    period: str
    total_trades: int
    direction_accuracy: float
    total_pnl: float


class EvaluationReport(BaseModel, frozen=True):
    total_evaluated: int
    total_unmatched: int
    direction_accuracy: float
    avg_entry_deviation_pct: float
    sl_hit_rate: float
    tp_hit_rate: float
    liquidation_rate: float
    by_symbol: list[SymbolStats]
    by_confidence: list[ConfidenceBucket]
    by_period: list[PeriodStats]
    evaluations: list[TradeEvaluation]


class PipelineEvaluator:
    def __init__(
        self,
        *,
        trade_repo: PaperTradeRepository,
        proposal_repo: TradeProposalRepository,
    ) -> None:
        self._trade_repo = trade_repo
        self._proposal_repo = proposal_repo

    def evaluate_single(self, trade_id: str) -> TradeEvaluation | None:
        """Evaluate a single closed trade against its proposal."""
        trade = self._trade_repo.get_by_trade_id(trade_id)
        if trade is None:
            return None

        proposal_rec = self._proposal_repo.get_by_proposal_id(trade.proposal_id)
        if proposal_rec is None:
            return None

        proposal = json.loads(proposal_rec.proposal_json)
        proposed_entry_price = proposal.get("entry", {}).get("price")
        raw_confidence = proposal.get("confidence", 0)
        # confidence may be 0-1 float or 1-100 int
        confidence = int(raw_confidence * 100) if raw_confidence <= 1.0 else int(raw_confidence)

        # Direction correct: proposal side matches profitable outcome
        direction_correct = trade.pnl > 0

        # Entry deviation
        entry_deviation_pct: float | None = None
        if proposed_entry_price is not None and proposed_entry_price > 0:
            entry_deviation_pct = (
                (trade.entry_price - proposed_entry_price) / proposed_entry_price * 100
            )

        return TradeEvaluation(
            trade_id=trade.trade_id,
            proposal_id=trade.proposal_id,
            symbol=trade.symbol,
            direction_correct=direction_correct,
            entry_deviation_pct=entry_deviation_pct,
            close_reason=trade.close_reason or "",
            confidence=confidence,
            pnl=trade.pnl,
        )

    def evaluate(
        self,
        *,
        symbol: str | None = None,
        since: datetime | None = None,
    ) -> EvaluationReport:
        """Evaluate all closed trades, optionally filtered."""
        closed_trades = self._trade_repo.get_all_closed()

        if symbol:
            closed_trades = [t for t in closed_trades if t.symbol == symbol]
        if since:
            closed_trades = [
                t for t in closed_trades
                if t.closed_at is not None and t.closed_at >= since
            ]

        evaluations: list[TradeEvaluation] = []
        unmatched = 0

        for trade in closed_trades:
            ev = self.evaluate_single(trade.trade_id)
            if ev is None:
                unmatched += 1
            else:
                evaluations.append(ev)

        return self._aggregate(evaluations, unmatched)

    def _aggregate(
        self, evaluations: list[TradeEvaluation], unmatched: int
    ) -> EvaluationReport:
        total = len(evaluations)
        if total == 0:
            return EvaluationReport(
                total_evaluated=0,
                total_unmatched=unmatched,
                direction_accuracy=0.0,
                avg_entry_deviation_pct=0.0,
                sl_hit_rate=0.0,
                tp_hit_rate=0.0,
                liquidation_rate=0.0,
                by_symbol=[],
                by_confidence=[],
                by_period=[],
                evaluations=[],
            )

        direction_correct = sum(1 for e in evaluations if e.direction_correct)
        deviations = [
            e.entry_deviation_pct for e in evaluations if e.entry_deviation_pct is not None
        ]
        sl_count = sum(1 for e in evaluations if e.close_reason == "sl")
        tp_count = sum(1 for e in evaluations if e.close_reason == "tp")
        liq_count = sum(1 for e in evaluations if e.close_reason == "liquidation")

        return EvaluationReport(
            total_evaluated=total,
            total_unmatched=unmatched,
            direction_accuracy=direction_correct / total,
            avg_entry_deviation_pct=(
                sum(deviations) / len(deviations) if deviations else 0.0
            ),
            sl_hit_rate=sl_count / total,
            tp_hit_rate=tp_count / total,
            liquidation_rate=liq_count / total,
            by_symbol=self._group_by_symbol(evaluations),
            by_confidence=self._group_by_confidence(evaluations),
            by_period=[],
            evaluations=evaluations,
        )

    def _group_by_symbol(self, evaluations: list[TradeEvaluation]) -> list[SymbolStats]:
        groups: dict[str, list[TradeEvaluation]] = defaultdict(list)
        for e in evaluations:
            groups[e.symbol].append(e)

        result: list[SymbolStats] = []
        for sym, evs in sorted(groups.items()):
            total = len(evs)
            deviations = [e.entry_deviation_pct for e in evs if e.entry_deviation_pct is not None]
            result.append(SymbolStats(
                symbol=sym,
                total_trades=total,
                direction_accuracy=sum(1 for e in evs if e.direction_correct) / total,
                avg_entry_deviation_pct=sum(deviations) / len(deviations) if deviations else 0.0,
                sl_hit_rate=sum(1 for e in evs if e.close_reason == "sl") / total,
                tp_hit_rate=sum(1 for e in evs if e.close_reason == "tp") / total,
                total_pnl=sum(e.pnl for e in evs),
            ))
        return result

    def _group_by_confidence(self, evaluations: list[TradeEvaluation]) -> list[ConfidenceBucket]:
        buckets: dict[str, list[TradeEvaluation]] = defaultdict(list)
        for e in evaluations:
            if e.confidence <= 33:
                buckets["low (1-33)"].append(e)
            elif e.confidence <= 66:
                buckets["mid (34-66)"].append(e)
            else:
                buckets["high (67-100)"].append(e)

        result: list[ConfidenceBucket] = []
        for bucket_name in ("low (1-33)", "mid (34-66)", "high (67-100)"):
            evs = buckets.get(bucket_name, [])
            if not evs:
                continue
            total = len(evs)
            result.append(ConfidenceBucket(
                bucket=bucket_name,
                total_trades=total,
                direction_accuracy=sum(1 for e in evs if e.direction_correct) / total,
                avg_pnl=sum(e.pnl for e in evs) / total,
            ))
        return result
