from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, Field


# --- Enums ---


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class Trend(StrEnum):
    UP = "up"
    DOWN = "down"
    RANGE = "range"


class VolatilityRegime(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# --- Sentiment ---


class KeyEvent(BaseModel, frozen=True):
    event: str
    impact: str
    source: str


class SentimentReport(BaseModel, frozen=True):
    sentiment_score: int = Field(ge=0, le=100)
    key_events: list[KeyEvent]
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


# --- Market ---


class KeyLevel(BaseModel, frozen=True):
    type: str
    price: float


class MarketInterpretation(BaseModel, frozen=True):
    trend: Trend
    volatility_regime: VolatilityRegime
    key_levels: list[KeyLevel]
    risk_flags: list[str]


# --- Trade Proposal ---


class EntryOrder(BaseModel, frozen=True):
    type: str  # "market" or "limit"
    price: float | None = None


class TradeProposal(BaseModel, frozen=True):
    proposal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    side: Side
    entry: EntryOrder
    position_size_risk_pct: float = Field(ge=0.0)
    stop_loss: float | None = None
    take_profit: list[float]
    time_horizon: str
    confidence: float = Field(ge=0.0, le=1.0)
    invalid_if: list[str]
    rationale: str
