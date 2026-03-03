from __future__ import annotations

from pydantic import BaseModel


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
