from __future__ import annotations

import json

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
