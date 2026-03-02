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


class Momentum(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# --- Common ---


class KeyLevel(BaseModel, frozen=True):
    type: str
    price: float


class TakeProfit(BaseModel, frozen=True):
    price: float
    close_pct: int = Field(ge=1, le=100)


# --- Technical Analysis ---


class TechnicalAnalysis(BaseModel, frozen=True):
    label: str  # "short_term" or "long_term"
    trend: Trend
    trend_strength: float = Field(ge=0.0)  # ADX value
    volatility_regime: VolatilityRegime
    volatility_pct: float = Field(ge=0.0, default=0.0)
    momentum: Momentum
    rsi: float = Field(ge=0.0, le=100.0)
    key_levels: list[KeyLevel]
    risk_flags: list[str]
    # Long-term only
    above_200w_ma: bool | None = None
    bull_support_band_status: str | None = None  # "above" | "within" | "below"


# --- Positioning ---


class PositioningAnalysis(BaseModel, frozen=True):
    funding_trend: str  # "rising" | "falling" | "stable"
    funding_extreme: bool
    oi_change_pct: float
    retail_bias: str  # "long" | "short" | "neutral"
    smart_money_bias: str  # "long" | "short" | "neutral"
    squeeze_risk: str  # "long_squeeze" | "short_squeeze" | "none"
    liquidity_assessment: str  # "thin" | "normal" | "deep"
    risk_flags: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


# --- Catalyst ---


class CatalystEvent(BaseModel, frozen=True):
    event: str
    time: str  # ISO timestamp or "ongoing"
    impact: str  # "high" | "medium" | "low"
    direction_bias: str  # "bullish" | "bearish" | "uncertain"


class CatalystReport(BaseModel, frozen=True):
    upcoming_events: list[CatalystEvent]
    active_events: list[CatalystEvent]
    risk_level: str  # "low" | "medium" | "high"
    recommendation: str  # "proceed" | "reduce_size" | "wait"
    confidence: float = Field(ge=0.0, le=1.0)


# --- Correlation ---


class CorrelationAnalysis(BaseModel, frozen=True):
    dxy_trend: str  # "strengthening" | "weakening" | "stable"
    dxy_impact: str  # "headwind" | "tailwind" | "neutral"
    sp500_regime: str  # "risk_on" | "risk_off" | "neutral"
    btc_dominance_trend: str  # "rising" | "falling" | "stable"
    cross_market_alignment: str  # "favorable" | "unfavorable" | "mixed"
    risk_flags: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


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
    take_profit: list[TakeProfit]
    suggested_leverage: int = Field(ge=1, le=50, default=10)
    time_horizon: str
    confidence: float = Field(ge=0.0, le=1.0)
    invalid_if: list[str]
    rationale: str
