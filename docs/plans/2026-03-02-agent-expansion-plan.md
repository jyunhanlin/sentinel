# Agent Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand the Sentinel pipeline from 3 agents (Sentiment, Market, Proposer) to 5 analysis agents (Technical×2, Positioning, Catalyst, Correlation) + Proposer, optimized for leveraged futures entry decisions.

**Architecture:** Replace Sentiment and Market agents with a single parameterized Technical agent (instantiated for short-term 4h and long-term 1d), add three new agents (Positioning, Catalyst, Correlation) that run in parallel via `asyncio.gather`, and update Proposer to consume all 4 analysis outputs. All data sources are free (CCXT/Binance API, Yahoo Finance, economic calendar).

**Tech Stack:** Python 3.12+, Pydantic v2 (frozen models), asyncio, CCXT, structlog, pytest-asyncio

**Design Doc:** `docs/plans/2026-03-02-agent-expansion-design.md`

---

## Task 1: Add new models for Technical, Positioning, Catalyst, Correlation

**Files:**
- Modify: `orchestrator/src/orchestrator/models.py`
- Test: `orchestrator/tests/unit/test_models_new.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_models_new.py
import pytest
from orchestrator.models import (
    Momentum,
    TechnicalAnalysis,
    PositioningAnalysis,
    CatalystEvent,
    CatalystReport,
    CorrelationAnalysis,
)


class TestTechnicalAnalysis:
    def test_create_short_term(self):
        ta = TechnicalAnalysis(
            label="short_term",
            trend="up",
            trend_strength=25.0,
            volatility_regime="medium",
            volatility_pct=2.3,
            momentum="bullish",
            rsi=65.0,
            key_levels=[],
            risk_flags=[],
        )
        assert ta.label == "short_term"
        assert ta.above_200w_ma is None
        assert ta.bull_support_band_status is None

    def test_create_long_term_with_macro(self):
        ta = TechnicalAnalysis(
            label="long_term",
            trend="up",
            trend_strength=30.0,
            volatility_regime="low",
            volatility_pct=1.2,
            momentum="bullish",
            rsi=58.0,
            key_levels=[],
            risk_flags=[],
            above_200w_ma=True,
            bull_support_band_status="above",
        )
        assert ta.above_200w_ma is True
        assert ta.bull_support_band_status == "above"

    def test_frozen(self):
        ta = TechnicalAnalysis(
            label="short_term", trend="up", trend_strength=25.0,
            volatility_regime="medium", volatility_pct=2.3,
            momentum="bullish", rsi=65.0, key_levels=[], risk_flags=[],
        )
        with pytest.raises(Exception):
            ta.trend = "down"


class TestPositioningAnalysis:
    def test_create(self):
        pa = PositioningAnalysis(
            funding_trend="rising",
            funding_extreme=False,
            oi_change_pct=5.2,
            retail_bias="long",
            smart_money_bias="short",
            squeeze_risk="long_squeeze",
            liquidity_assessment="normal",
            risk_flags=["funding_elevated"],
            confidence=0.7,
        )
        assert pa.funding_trend == "rising"
        assert pa.squeeze_risk == "long_squeeze"

    def test_frozen(self):
        pa = PositioningAnalysis(
            funding_trend="stable", funding_extreme=False, oi_change_pct=0.0,
            retail_bias="neutral", smart_money_bias="neutral", squeeze_risk="none",
            liquidity_assessment="normal", risk_flags=[], confidence=0.5,
        )
        with pytest.raises(Exception):
            pa.funding_trend = "falling"


class TestCatalystReport:
    def test_create_with_events(self):
        event = CatalystEvent(
            event="FOMC Rate Decision",
            time="2026-03-15T18:00:00Z",
            impact="high",
            direction_bias="uncertain",
        )
        report = CatalystReport(
            upcoming_events=[event],
            active_events=[],
            risk_level="high",
            recommendation="wait",
            confidence=0.8,
        )
        assert len(report.upcoming_events) == 1
        assert report.recommendation == "wait"

    def test_frozen(self):
        report = CatalystReport(
            upcoming_events=[], active_events=[],
            risk_level="low", recommendation="proceed", confidence=0.6,
        )
        with pytest.raises(Exception):
            report.risk_level = "high"


class TestCorrelationAnalysis:
    def test_create(self):
        ca = CorrelationAnalysis(
            dxy_trend="strengthening",
            dxy_impact="headwind",
            sp500_regime="risk_off",
            btc_dominance_trend="rising",
            cross_market_alignment="unfavorable",
            risk_flags=["dxy_headwind"],
            confidence=0.6,
        )
        assert ca.cross_market_alignment == "unfavorable"

    def test_frozen(self):
        ca = CorrelationAnalysis(
            dxy_trend="stable", dxy_impact="neutral", sp500_regime="neutral",
            btc_dominance_trend="stable", cross_market_alignment="mixed",
            risk_flags=[], confidence=0.5,
        )
        with pytest.raises(Exception):
            ca.dxy_trend = "weakening"
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_models_new.py -v`
Expected: FAIL with `ImportError: cannot import name 'TechnicalAnalysis'`

**Step 3: Write minimal implementation**

Add the following to `orchestrator/src/orchestrator/models.py` after the existing `MarketInterpretation` class:

```python
# --- Enums (add new) ---

class Momentum(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# --- Technical Analysis (replaces MarketInterpretation for new pipeline) ---

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
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_models_new.py -v`
Expected: PASS (all 8 tests)

**Step 5: Run full test suite**

Run: `cd orchestrator && uv run pytest -v`
Expected: All existing tests still pass (no breaking changes — old models preserved)

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/models.py orchestrator/tests/unit/test_models_new.py
git commit -m "feat: add Pydantic models for Technical, Positioning, Catalyst, Correlation agents"
```

---

## Task 2: Create Technical agent (replaces Market agent)

**Files:**
- Create: `orchestrator/src/orchestrator/agents/technical.py`
- Test: `orchestrator/tests/unit/test_agent_technical.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_agent_technical.py
from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.technical import TechnicalAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import TechnicalAnalysis, Trend, Momentum


def make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT:USDT",
        timeframe="4h",
        current_price=95200.0,
        volume_24h=1_000_000.0,
        funding_rate=0.0001,
        ohlcv=[
            [1700000000000, 94000.0, 95500.0, 93500.0, 95200.0, 1000.0],
            [1700003600000, 95200.0, 96000.0, 95000.0, 95800.0, 800.0],
        ],
    )


VALID_JSON = (
    '```json\n'
    '{"label": "short_term", "trend": "up", "trend_strength": 28.5, '
    '"volatility_regime": "medium", "volatility_pct": 2.3, '
    '"momentum": "bullish", "rsi": 62.0, '
    '"key_levels": [{"type": "support", "price": 93000}], '
    '"risk_flags": []}\n```'
)


class TestTechnicalAgent:
    @pytest.mark.asyncio
    async def test_prompt_contains_skill_name_and_label(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=VALID_JSON, model="test",
            input_tokens=200, output_tokens=100, latency_ms=1000,
        )

        agent = TechnicalAgent(client=mock_client, label="short_term", candle_count=50)
        await agent.analyze(snapshot=make_snapshot())

        messages = mock_client.call.call_args[0][0]
        prompt = messages[0]["content"]
        assert "technical" in prompt.lower()
        assert "skill" in prompt.lower()
        assert "short_term" in prompt

    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=VALID_JSON, model="test",
            input_tokens=200, output_tokens=100, latency_ms=1000,
        )

        agent = TechnicalAgent(client=mock_client, label="short_term", candle_count=50)
        result = await agent.analyze(snapshot=make_snapshot())

        assert isinstance(result.output, TechnicalAnalysis)
        assert result.output.trend == Trend.UP
        assert result.output.momentum == Momentum.BULLISH
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_long_term_includes_macro_data(self):
        mock_client = AsyncMock(spec=LLMClient)
        long_json = (
            '```json\n'
            '{"label": "long_term", "trend": "up", "trend_strength": 30.0, '
            '"volatility_regime": "low", "volatility_pct": 1.2, '
            '"momentum": "bullish", "rsi": 58.0, '
            '"key_levels": [], "risk_flags": [], '
            '"above_200w_ma": true, "bull_support_band_status": "above"}\n```'
        )
        mock_client.call.return_value = LLMCallResult(
            content=long_json, model="test",
            input_tokens=200, output_tokens=100, latency_ms=1000,
        )

        agent = TechnicalAgent(client=mock_client, label="long_term", candle_count=30)
        result = await agent.analyze(
            snapshot=make_snapshot(),
            macro_data={"ma_200w": 42000.0, "bull_support_upper": 68000.0, "bull_support_lower": 65000.0},
        )

        prompt = mock_client.call.call_args[0][0][0]["content"]
        assert "200W MA" in prompt
        assert "42000" in prompt

    @pytest.mark.asyncio
    async def test_degrade_returns_default(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken", model="test",
            input_tokens=100, output_tokens=50, latency_ms=500,
        )

        agent = TechnicalAgent(client=mock_client, label="short_term", candle_count=50, max_retries=0)
        result = await agent.analyze(snapshot=make_snapshot())

        assert result.degraded is True
        assert result.output.trend == Trend.RANGE
        assert result.output.label == "short_term"
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_technical.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.agents.technical'`

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/agents/technical.py
from __future__ import annotations

from typing import Any

from orchestrator.agents.base import BaseAgent
from orchestrator.agents.utils import summarize_ohlcv
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.models import (
    Momentum,
    TechnicalAnalysis,
    Trend,
    VolatilityRegime,
)


class TechnicalAgent(BaseAgent[TechnicalAnalysis]):
    output_model = TechnicalAnalysis
    _skill_name = "technical"

    def __init__(
        self,
        client,
        *,
        label: str = "short_term",
        candle_count: int = 50,
        max_retries: int = 1,
    ) -> None:
        super().__init__(client, max_retries=max_retries)
        self._label = label
        self._candle_count = candle_count

    def _build_prompt(self, **kwargs) -> str:
        snapshot: MarketSnapshot = kwargs["snapshot"]
        macro_data: dict[str, Any] | None = kwargs.get("macro_data")

        ohlcv_summary = summarize_ohlcv(snapshot.ohlcv, max_candles=self._candle_count)

        data = (
            f"Symbol: {snapshot.symbol}\n"
            f"Current Price: {snapshot.current_price}\n"
            f"24h Volume: {snapshot.volume_24h:,.0f}\n"
            f"Funding Rate: {snapshot.funding_rate:.6f}\n"
            f"Timeframe: {snapshot.timeframe}\n"
            f"Analysis Label: {self._label}\n\n"
            f"OHLCV Data ({len(snapshot.ohlcv)} candles):\n{ohlcv_summary}"
        )

        if macro_data:
            data += (
                f"\n\n=== Macro Indicators ===\n"
                f"200W MA: {macro_data['ma_200w']:.0f}\n"
                f"Bull Support Band: {macro_data['bull_support_upper']:.0f} - "
                f"{macro_data['bull_support_lower']:.0f}"
            )

        return (
            f"Use the {self._skill_name} skill.\n\n"
            f"=== Market Data ===\n{data}"
        )

    def _get_default_output(self) -> TechnicalAnalysis:
        return TechnicalAnalysis(
            label=self._label,
            trend=Trend.RANGE,
            trend_strength=0.0,
            volatility_regime=VolatilityRegime.MEDIUM,
            volatility_pct=0.0,
            momentum=Momentum.NEUTRAL,
            rsi=50.0,
            key_levels=[],
            risk_flags=["analysis_degraded"],
        )
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_technical.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/agents/technical.py orchestrator/tests/unit/test_agent_technical.py
git commit -m "feat: add TechnicalAgent with parameterized label and candle count"
```

---

## Task 3: Create Technical SKILL.md

**Files:**
- Create: `.claude/skills/technical/SKILL.md`

**Step 1: Write the skill file**

```markdown
---
name: technical
description: Crypto technical analysis with indicators — feeds into trade proposer
---

# Crypto Technical Analyst

## Context

You are part of a crypto futures trading pipeline. Your job is to analyze OHLCV data
using technical indicators to determine market structure, trend strength, and momentum.
Your output feeds directly into the **proposer** skill, which uses your analysis
to generate trade proposals for leveraged futures positions.

You are analyzing the **{label}** timeframe. Adjust your interpretation accordingly:
- **short_term**: focus on momentum, immediate price action, and short-duration setups
- **long_term**: focus on structural trends, macro context, and longer-duration setups

## Input Description

| Field | Type | Meaning |
|-------|------|---------|
| symbol | string | Trading pair (e.g. BTC/USDT:USDT) |
| current_price | float | Latest price |
| volume_24h | float | 24-hour trading volume in quote currency |
| funding_rate | float | Perpetual futures funding rate (8h) |
| timeframe | string | Candle timeframe (e.g. 4h, 1d) |
| label | string | "short_term" or "long_term" |
| ohlcv | table | Recent OHLCV candles: O, H, L, C, V |

### Macro Indicators (long_term only)

If provided:
| Field | Type | Meaning |
|-------|------|---------|
| 200W MA | float | 200-week simple moving average — macro bull/bear boundary |
| Bull Support Band | float range | 20W SMA to 21W EMA — bull market pullback zone |

## Methodology

Think through each step before producing output.

### Step 1: Trend Identification
- Compare current price to the first candle's open
- Count green vs red candles
- Look for higher highs + higher lows (uptrend) or lower highs + lower lows (downtrend)
- If price oscillating within a band without clear direction → range

### Step 2: Trend Strength (ADX)
Estimate ADX(14) from the OHLCV data:
- Calculate +DM and -DM for each candle
- Smooth over 14 periods → +DI and -DI
- ADX = smoothed average of |+DI - -DI| / (+DI + -DI) × 100

| ADX | Interpretation |
|-----|---------------|
| < 20 | Weak/no trend — ranging market, dangerous for leverage |
| 20-40 | Moderate trend |
| 40-60 | Strong trend |
| > 60 | Very strong trend |

### Step 3: Momentum (RSI + MACD)
**RSI(14):**
- Calculate average gains and losses over 14 periods
- RSI = 100 - (100 / (1 + avg_gain/avg_loss))

| RSI | Meaning |
|-----|---------|
| < 30 | Oversold — potential reversal up |
| 30-70 | Neutral range |
| > 70 | Overbought — potential reversal down |

**MACD(12, 26, 9):**
- MACD line = EMA(12) - EMA(26)
- Signal line = EMA(9) of MACD
- Histogram = MACD - Signal
- Bullish: histogram positive and growing
- Bearish: histogram negative and growing
- Divergence: price makes new high but MACD doesn't (bearish) or vice versa

**Synthesize momentum:**
- RSI > 50 + MACD histogram positive → "bullish"
- RSI < 50 + MACD histogram negative → "bearish"
- Mixed signals → "neutral"

### Step 4: Volatility Assessment
**ATR(14):** for each of the last 14 candles, compute
True Range = max(high - low, |high - prev_close|, |low - prev_close|).
Average these, divide by current price, multiply by 100 → volatility_pct.

| volatility_pct | Regime |
|----------------|--------|
| < 1.5% | low |
| 1.5% - 3.5% | medium |
| > 3.5% | high |

**Bollinger Bands(20, 2):**
- Middle = SMA(20)
- Upper/Lower = Middle ± 2 × StdDev(20)
- Price above upper band → overextended, potential reversal
- Price below lower band → oversold, potential bounce
- Band squeeze (narrow bands) → expect breakout

### Step 5: Moving Averages
- Calculate EMA(20) and EMA(50)
- EMA(20) > EMA(50) → bullish structure
- EMA(20) < EMA(50) → bearish structure
- Price relative to EMAs confirms trend direction

### Step 6: Key Levels
- **Support**: price levels where multiple candle lows cluster or where price bounced
- **Resistance**: price levels where multiple candle highs cluster or where price rejected
- Round numbers near current price are psychologically significant
- Only include levels within ±5% of current price
- Maximum 3 support + 3 resistance levels

### Step 7: K-Line Patterns
- Long wicks on top → selling pressure / rejection
- Long wicks on bottom → buying pressure / absorption
- Consecutive green closes → bullish momentum
- Consecutive red closes → bearish momentum
- Doji/spinning tops → indecision

### Step 8: Risk Flags
Flag conditions that increase trading risk:

| Flag | Trigger |
|------|---------|
| `funding_elevated` | abs(funding_rate) > 0.05% |
| `volume_declining` | last 3 candles volume each lower than previous |
| `high_volatility` | volatility_pct > 5% |
| `near_key_level` | price within 0.3% of support or resistance |
| `trend_exhaustion` | >8 consecutive same-color candles |
| `overbought` | RSI > 75 |
| `oversold` | RSI < 25 |
| `bollinger_squeeze` | band width < 50% of 20-period average band width |
| `macd_divergence` | price and MACD moving in opposite directions |

### Step 9: Macro Context (long_term only)
If 200W MA and Bull Support Band are provided:
- `above_200w_ma`: is current price above 200W MA?
- `bull_support_band_status`:
  - "above" — price above upper band (healthy bull)
  - "within" — price inside band (pullback zone, potential buy)
  - "below" — price below lower band (bearish, caution)

## Output

After your analysis, output a single fenced JSON block:

```json
{
  "label": "short_term" | "long_term",
  "trend": "up" | "down" | "range",
  "trend_strength": <float (ADX value)>,
  "volatility_regime": "low" | "medium" | "high",
  "volatility_pct": <float>,
  "momentum": "bullish" | "bearish" | "neutral",
  "rsi": <float 0-100>,
  "key_levels": [{"type": "support" | "resistance", "price": <float>}],
  "risk_flags": ["<flag_name>"],
  "above_200w_ma": <bool | null>,
  "bull_support_band_status": "above" | "within" | "below" | null
}
```

Field notes:
- Set `label` to match the Analysis Label from input
- `above_200w_ma` and `bull_support_band_status`: set to null if no macro data provided
- `risk_flags`: empty list if no flags triggered

## Historical Context

If a "Historical Context" section is provided in the input data, reference past market
conditions and how they resolved to inform your current analysis.
```

**Step 2: Commit**

```bash
git add .claude/skills/technical/SKILL.md
git commit -m "feat: add Technical skill with ADX, RSI, MACD, Bollinger Bands, EMA"
```

---

## Task 4: Create Positioning agent + SKILL.md

**Files:**
- Create: `orchestrator/src/orchestrator/agents/positioning.py`
- Create: `.claude/skills/positioning/SKILL.md`
- Test: `orchestrator/tests/unit/test_agent_positioning.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_agent_positioning.py
from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.positioning import PositioningAgent
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import PositioningAnalysis


VALID_JSON = (
    '```json\n'
    '{"funding_trend": "rising", "funding_extreme": false, '
    '"oi_change_pct": 3.5, "retail_bias": "long", '
    '"smart_money_bias": "short", "squeeze_risk": "long_squeeze", '
    '"liquidity_assessment": "normal", '
    '"risk_flags": ["funding_elevated"], "confidence": 0.7}\n```'
)


class TestPositioningAgent:
    @pytest.mark.asyncio
    async def test_prompt_contains_skill_and_data(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=VALID_JSON, model="test",
            input_tokens=200, output_tokens=100, latency_ms=1000,
        )

        agent = PositioningAgent(client=mock_client)
        await agent.analyze(
            symbol="BTC/USDT:USDT",
            current_price=95200.0,
            funding_rate_history=[0.0001, 0.0002, 0.0003],
            open_interest=5_000_000_000.0,
            oi_change_pct=3.5,
            long_short_ratio=1.2,
            top_trader_long_short_ratio=0.9,
            order_book_summary={"bid_depth": 100.0, "ask_depth": 120.0},
        )

        messages = mock_client.call.call_args[0][0]
        prompt = messages[0]["content"]
        assert "positioning" in prompt.lower()
        assert "skill" in prompt.lower()
        assert "BTC/USDT:USDT" in prompt

    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=VALID_JSON, model="test",
            input_tokens=200, output_tokens=100, latency_ms=1000,
        )

        agent = PositioningAgent(client=mock_client)
        result = await agent.analyze(
            symbol="BTC/USDT:USDT",
            current_price=95200.0,
            funding_rate_history=[0.0001],
            open_interest=5_000_000_000.0,
            oi_change_pct=3.5,
            long_short_ratio=1.2,
            top_trader_long_short_ratio=0.9,
            order_book_summary={"bid_depth": 100.0, "ask_depth": 120.0},
        )

        assert isinstance(result.output, PositioningAnalysis)
        assert result.output.squeeze_risk == "long_squeeze"
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_degrade_returns_default(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken", model="test",
            input_tokens=100, output_tokens=50, latency_ms=500,
        )

        agent = PositioningAgent(client=mock_client, max_retries=0)
        result = await agent.analyze(
            symbol="BTC/USDT:USDT",
            current_price=95200.0,
            funding_rate_history=[],
            open_interest=0.0,
            oi_change_pct=0.0,
            long_short_ratio=1.0,
            top_trader_long_short_ratio=1.0,
            order_book_summary={"bid_depth": 0.0, "ask_depth": 0.0},
        )

        assert result.degraded is True
        assert result.output.funding_trend == "stable"
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_positioning.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the agent**

```python
# orchestrator/src/orchestrator/agents/positioning.py
from __future__ import annotations

from typing import Any

from orchestrator.agents.base import BaseAgent
from orchestrator.models import PositioningAnalysis


class PositioningAgent(BaseAgent[PositioningAnalysis]):
    output_model = PositioningAnalysis
    _skill_name = "positioning"

    def _build_prompt(self, **kwargs) -> str:
        symbol: str = kwargs["symbol"]
        current_price: float = kwargs["current_price"]
        funding_history: list[float] = kwargs["funding_rate_history"]
        open_interest: float = kwargs["open_interest"]
        oi_change_pct: float = kwargs["oi_change_pct"]
        ls_ratio: float = kwargs["long_short_ratio"]
        top_ls_ratio: float = kwargs["top_trader_long_short_ratio"]
        order_book: dict[str, Any] = kwargs["order_book_summary"]

        funding_str = ", ".join(f"{r:.6f}" for r in funding_history[-10:])

        data = (
            f"Symbol: {symbol}\n"
            f"Current Price: {current_price}\n\n"
            f"=== Funding Rate History (last {len(funding_history[-10:])} periods) ===\n"
            f"{funding_str}\n\n"
            f"=== Open Interest ===\n"
            f"Current OI: {open_interest:,.0f}\n"
            f"OI Change: {oi_change_pct:+.1f}%\n\n"
            f"=== Long/Short Ratios ===\n"
            f"Retail L/S Ratio: {ls_ratio:.2f}\n"
            f"Top Trader L/S Ratio: {top_ls_ratio:.2f}\n\n"
            f"=== Order Book ===\n"
            f"Bid Depth: {order_book.get('bid_depth', 0):.0f}\n"
            f"Ask Depth: {order_book.get('ask_depth', 0):.0f}"
        )

        return (
            f"Use the {self._skill_name} skill.\n\n"
            f"=== Positioning Data ===\n{data}"
        )

    def _get_default_output(self) -> PositioningAnalysis:
        return PositioningAnalysis(
            funding_trend="stable",
            funding_extreme=False,
            oi_change_pct=0.0,
            retail_bias="neutral",
            smart_money_bias="neutral",
            squeeze_risk="none",
            liquidity_assessment="normal",
            risk_flags=["analysis_degraded"],
            confidence=0.1,
        )
```

**Step 4: Write the SKILL.md**

```markdown
---
name: positioning
description: Crypto futures positioning and order flow analysis — feeds into trade proposer
---

# Crypto Positioning Analyst

## Context

You are part of a crypto futures trading pipeline. Your job is to analyze derivatives
positioning data — funding rates, open interest, long/short ratios, and order book depth —
to understand how market participants are positioned. Your output feeds directly into
the **proposer** skill, which uses your analysis to assess squeeze risk, crowding,
and optimal leverage for trade proposals.

## Input Description

| Field | Type | Meaning |
|-------|------|---------|
| symbol | string | Trading pair (e.g. BTC/USDT:USDT) |
| current_price | float | Latest price |
| funding_rate_history | float[] | Recent 8h funding rates (newest last) |
| open_interest | float | Current aggregate open interest in USD |
| oi_change_pct | float | OI change % over recent period |
| long_short_ratio | float | Retail accounts long/short ratio (>1 = more longs) |
| top_trader_long_short_ratio | float | Top trader long/short ratio (proxy for smart money) |
| order_book | bid_depth, ask_depth | Aggregated bid/ask depth near current price |

## Methodology

Think through each step before producing output.

### Step 1: Funding Rate Analysis
- Look at the trend across the history, not just the latest value
- Rising funding → increasing long bias
- Falling funding → increasing short bias
- Stable funding → no directional shift
- Extreme: abs(latest funding) > 0.05% → `funding_extreme = true`

### Step 2: Open Interest Analysis
- Rising OI + rising price → new longs entering (bullish conviction)
- Rising OI + falling price → new shorts entering (bearish conviction)
- Falling OI + rising price → short covering (weak rally)
- Falling OI + falling price → long liquidation (capitulation)

### Step 3: Long/Short Ratio Interpretation
**Retail (long_short_ratio):**
- > 1.5 → retail heavily long → contrarian bearish signal
- < 0.7 → retail heavily short → contrarian bullish signal
- 0.7-1.5 → no extreme → neutral

**Top Traders (top_trader_long_short_ratio):**
- > 1.2 → smart money leaning long → follow (bullish)
- < 0.8 → smart money leaning short → follow (bearish)
- 0.8-1.2 → no strong lean → neutral

### Step 4: Squeeze Risk Assessment
- Retail long + OI rising + funding rising → long squeeze risk (too crowded long)
- Retail short + OI rising + funding falling → short squeeze risk (too crowded short)
- If retail and smart money disagree → squeeze more likely on retail side

### Step 5: Liquidity Assessment
- Compare bid_depth vs ask_depth
- If both are low relative to typical → "thin" (dangerous for leverage, slippage risk)
- If balanced and substantial → "normal"
- If very deep → "deep" (safe for larger positions)

### Step 6: Risk Flags
| Flag | Trigger |
|------|---------|
| `funding_elevated` | abs(funding) > 0.05% |
| `oi_divergence` | OI direction contradicts price direction |
| `crowded_long` | retail L/S ratio > 2.0 |
| `crowded_short` | retail L/S ratio < 0.5 |
| `smart_money_disagrees` | retail and top trader ratios point opposite directions |
| `thin_liquidity` | order book depth is thin |

## Output

After your analysis, output a single fenced JSON block:

```json
{
  "funding_trend": "rising" | "falling" | "stable",
  "funding_extreme": <bool>,
  "oi_change_pct": <float>,
  "retail_bias": "long" | "short" | "neutral",
  "smart_money_bias": "long" | "short" | "neutral",
  "squeeze_risk": "long_squeeze" | "short_squeeze" | "none",
  "liquidity_assessment": "thin" | "normal" | "deep",
  "risk_flags": ["<flag_name>"],
  "confidence": <float 0.0-1.0>
}
```

## Historical Context

If a "Historical Context" section is provided in the input data, reference past
positioning conditions and how they resolved (e.g., did a crowded long position
lead to a squeeze?) to inform your current analysis.
```

**Step 5: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_positioning.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/agents/positioning.py orchestrator/tests/unit/test_agent_positioning.py .claude/skills/positioning/SKILL.md
git commit -m "feat: add Positioning agent and skill for order flow analysis"
```

---

## Task 5: Create Catalyst agent + SKILL.md

**Files:**
- Create: `orchestrator/src/orchestrator/agents/catalyst.py`
- Create: `.claude/skills/catalyst/SKILL.md`
- Test: `orchestrator/tests/unit/test_agent_catalyst.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_agent_catalyst.py
from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.catalyst import CatalystAgent
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import CatalystReport


VALID_JSON = (
    '```json\n'
    '{"upcoming_events": [{"event": "FOMC Rate Decision", '
    '"time": "2026-03-15T18:00:00Z", "impact": "high", '
    '"direction_bias": "uncertain"}], '
    '"active_events": [], "risk_level": "high", '
    '"recommendation": "wait", "confidence": 0.8}\n```'
)


class TestCatalystAgent:
    @pytest.mark.asyncio
    async def test_prompt_contains_skill_and_data(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=VALID_JSON, model="test",
            input_tokens=200, output_tokens=100, latency_ms=1000,
        )

        agent = CatalystAgent(client=mock_client)
        await agent.analyze(
            symbol="BTC/USDT:USDT",
            current_price=95200.0,
            economic_calendar=[{"event": "FOMC", "time": "2026-03-15T18:00:00Z", "impact": "high"}],
            exchange_announcements=["Binance will delist TOKEN/USDT on 2026-03-20"],
        )

        messages = mock_client.call.call_args[0][0]
        prompt = messages[0]["content"]
        assert "catalyst" in prompt.lower()
        assert "FOMC" in prompt

    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=VALID_JSON, model="test",
            input_tokens=200, output_tokens=100, latency_ms=1000,
        )

        agent = CatalystAgent(client=mock_client)
        result = await agent.analyze(
            symbol="BTC/USDT:USDT",
            current_price=95200.0,
            economic_calendar=[],
            exchange_announcements=[],
        )

        assert isinstance(result.output, CatalystReport)
        assert result.output.recommendation == "wait"
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_degrade_returns_default(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken", model="test",
            input_tokens=100, output_tokens=50, latency_ms=500,
        )

        agent = CatalystAgent(client=mock_client, max_retries=0)
        result = await agent.analyze(
            symbol="BTC/USDT:USDT",
            current_price=95200.0,
            economic_calendar=[],
            exchange_announcements=[],
        )

        assert result.degraded is True
        assert result.output.recommendation == "proceed"
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_catalyst.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the agent**

```python
# orchestrator/src/orchestrator/agents/catalyst.py
from __future__ import annotations

from typing import Any

from orchestrator.agents.base import BaseAgent
from orchestrator.models import CatalystReport


class CatalystAgent(BaseAgent[CatalystReport]):
    output_model = CatalystReport
    _skill_name = "catalyst"

    def _build_prompt(self, **kwargs) -> str:
        symbol: str = kwargs["symbol"]
        current_price: float = kwargs["current_price"]
        calendar: list[dict[str, Any]] = kwargs["economic_calendar"]
        announcements: list[str] = kwargs["exchange_announcements"]

        calendar_str = ""
        for entry in calendar:
            calendar_str += f"- {entry.get('event', 'Unknown')} | {entry.get('time', 'TBD')} | Impact: {entry.get('impact', 'unknown')}\n"
        if not calendar_str:
            calendar_str = "No upcoming events\n"

        announcements_str = ""
        for ann in announcements:
            announcements_str += f"- {ann}\n"
        if not announcements_str:
            announcements_str = "No recent announcements\n"

        data = (
            f"Symbol: {symbol}\n"
            f"Current Price: {current_price}\n\n"
            f"=== Economic Calendar (next 48h) ===\n"
            f"{calendar_str}\n"
            f"=== Exchange Announcements ===\n"
            f"{announcements_str}"
        )

        return (
            f"Use the {self._skill_name} skill.\n\n"
            f"=== Catalyst Data ===\n{data}"
        )

    def _get_default_output(self) -> CatalystReport:
        return CatalystReport(
            upcoming_events=[],
            active_events=[],
            risk_level="low",
            recommendation="proceed",
            confidence=0.1,
        )
```

**Step 4: Write the SKILL.md**

Create `.claude/skills/catalyst/SKILL.md` with methodology for assessing event impact on leveraged trading:
- Economic events: categorize by impact level, assess timing relative to trade horizon
- Exchange announcements: flag relevant listing/delisting/maintenance events
- Output recommendation: proceed / reduce_size / wait

(Full SKILL.md content follows same pattern as positioning — see implementation for exact text.)

**Step 5: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_catalyst.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/agents/catalyst.py orchestrator/tests/unit/test_agent_catalyst.py .claude/skills/catalyst/SKILL.md
git commit -m "feat: add Catalyst agent and skill for news/event analysis"
```

---

## Task 6: Create Correlation agent + SKILL.md

**Files:**
- Create: `orchestrator/src/orchestrator/agents/correlation.py`
- Create: `.claude/skills/correlation/SKILL.md`
- Test: `orchestrator/tests/unit/test_agent_correlation.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_agent_correlation.py
from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.correlation import CorrelationAgent
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import CorrelationAnalysis


VALID_JSON = (
    '```json\n'
    '{"dxy_trend": "strengthening", "dxy_impact": "headwind", '
    '"sp500_regime": "risk_off", "btc_dominance_trend": "rising", '
    '"cross_market_alignment": "unfavorable", '
    '"risk_flags": ["dxy_headwind"], "confidence": 0.7}\n```'
)


class TestCorrelationAgent:
    @pytest.mark.asyncio
    async def test_prompt_contains_skill_and_data(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=VALID_JSON, model="test",
            input_tokens=200, output_tokens=100, latency_ms=1000,
        )

        agent = CorrelationAgent(client=mock_client)
        await agent.analyze(
            symbol="BTC/USDT:USDT",
            dxy_data={"current": 104.5, "change_pct": 0.3, "trend_5d": [103.8, 104.0, 104.2, 104.3, 104.5]},
            sp500_data={"current": 5800.0, "change_pct": -1.2, "trend_5d": [5900, 5880, 5850, 5820, 5800]},
            btc_dominance={"current": 54.2, "change_7d": 1.5},
        )

        messages = mock_client.call.call_args[0][0]
        prompt = messages[0]["content"]
        assert "correlation" in prompt.lower()
        assert "DXY" in prompt

    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=VALID_JSON, model="test",
            input_tokens=200, output_tokens=100, latency_ms=1000,
        )

        agent = CorrelationAgent(client=mock_client)
        result = await agent.analyze(
            symbol="BTC/USDT:USDT",
            dxy_data={"current": 104.5, "change_pct": 0.3, "trend_5d": []},
            sp500_data={"current": 5800.0, "change_pct": -1.2, "trend_5d": []},
            btc_dominance={"current": 54.2, "change_7d": 1.5},
        )

        assert isinstance(result.output, CorrelationAnalysis)
        assert result.output.dxy_impact == "headwind"
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_degrade_returns_default(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken", model="test",
            input_tokens=100, output_tokens=50, latency_ms=500,
        )

        agent = CorrelationAgent(client=mock_client, max_retries=0)
        result = await agent.analyze(
            symbol="BTC/USDT:USDT",
            dxy_data={"current": 0, "change_pct": 0, "trend_5d": []},
            sp500_data={"current": 0, "change_pct": 0, "trend_5d": []},
            btc_dominance={"current": 0, "change_7d": 0},
        )

        assert result.degraded is True
        assert result.output.cross_market_alignment == "mixed"
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_correlation.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the agent**

```python
# orchestrator/src/orchestrator/agents/correlation.py
from __future__ import annotations

from typing import Any

from orchestrator.agents.base import BaseAgent
from orchestrator.models import CorrelationAnalysis


class CorrelationAgent(BaseAgent[CorrelationAnalysis]):
    output_model = CorrelationAnalysis
    _skill_name = "correlation"

    def _build_prompt(self, **kwargs) -> str:
        symbol: str = kwargs["symbol"]
        dxy: dict[str, Any] = kwargs["dxy_data"]
        sp500: dict[str, Any] = kwargs["sp500_data"]
        btc_dom: dict[str, Any] = kwargs["btc_dominance"]

        dxy_trend_str = ", ".join(f"{v:.1f}" for v in dxy.get("trend_5d", []))
        sp500_trend_str = ", ".join(f"{v:.0f}" for v in sp500.get("trend_5d", []))

        data = (
            f"Symbol: {symbol}\n\n"
            f"=== DXY (US Dollar Index) ===\n"
            f"Current: {dxy.get('current', 0):.1f}\n"
            f"Change: {dxy.get('change_pct', 0):+.2f}%\n"
            f"5-Day Trend: {dxy_trend_str or 'N/A'}\n\n"
            f"=== S&P 500 ===\n"
            f"Current: {sp500.get('current', 0):.0f}\n"
            f"Change: {sp500.get('change_pct', 0):+.2f}%\n"
            f"5-Day Trend: {sp500_trend_str or 'N/A'}\n\n"
            f"=== BTC Dominance ===\n"
            f"Current: {btc_dom.get('current', 0):.1f}%\n"
            f"7-Day Change: {btc_dom.get('change_7d', 0):+.1f}%"
        )

        return (
            f"Use the {self._skill_name} skill.\n\n"
            f"=== Cross-Market Data ===\n{data}"
        )

    def _get_default_output(self) -> CorrelationAnalysis:
        return CorrelationAnalysis(
            dxy_trend="stable",
            dxy_impact="neutral",
            sp500_regime="neutral",
            btc_dominance_trend="stable",
            cross_market_alignment="mixed",
            risk_flags=["analysis_degraded"],
            confidence=0.1,
        )
```

**Step 4: Write the SKILL.md**

Create `.claude/skills/correlation/SKILL.md` with methodology for cross-market analysis:
- DXY interpretation: strength = headwind for crypto, weakness = tailwind
- S&P 500: risk-on/off regime detection
- BTC dominance: capital rotation signals
- Synthesize alignment assessment

(Full SKILL.md follows same structure as other skills.)

**Step 5: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_correlation.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/agents/correlation.py orchestrator/tests/unit/test_agent_correlation.py .claude/skills/correlation/SKILL.md
git commit -m "feat: add Correlation agent and skill for cross-market analysis"
```

---

## Task 7: Extend DataFetcher for new data sources

**Files:**
- Modify: `orchestrator/src/orchestrator/exchange/data_fetcher.py`
- Modify: `orchestrator/src/orchestrator/exchange/client.py`
- Test: `orchestrator/tests/unit/test_data_fetcher_extended.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_data_fetcher_extended.py
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.exchange.client import ExchangeClient
from orchestrator.exchange.data_fetcher import DataFetcher


class TestDataFetcherExtended:
    @pytest.mark.asyncio
    async def test_fetch_positioning_data(self):
        mock_client = AsyncMock(spec=ExchangeClient)
        mock_client.fetch_funding_rate_history.return_value = [0.0001, 0.0002, 0.0003]
        mock_client.fetch_open_interest.return_value = 5_000_000_000.0
        mock_client.fetch_long_short_ratio.return_value = 1.2
        mock_client.fetch_top_trader_long_short_ratio.return_value = 0.9
        mock_client.fetch_order_book.return_value = {"bids": [[95000, 10]], "asks": [[95200, 12]]}

        fetcher = DataFetcher(mock_client)
        result = await fetcher.fetch_positioning_data("BTC/USDT:USDT")

        assert result["funding_rate_history"] == [0.0001, 0.0002, 0.0003]
        assert result["open_interest"] == 5_000_000_000.0
        assert result["long_short_ratio"] == 1.2

    @pytest.mark.asyncio
    async def test_fetch_macro_indicators(self):
        mock_client = AsyncMock(spec=ExchangeClient)
        mock_client.fetch_ohlcv.return_value = [
            [i * 604800000, 40000 + i * 100, 41000 + i * 100, 39000 + i * 100, 40500 + i * 100, 1000]
            for i in range(210)
        ]

        fetcher = DataFetcher(mock_client)
        result = await fetcher.fetch_macro_indicators("BTC/USDT:USDT")

        assert "ma_200w" in result
        assert "bull_support_upper" in result
        assert "bull_support_lower" in result
        assert result["ma_200w"] > 0
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_data_fetcher_extended.py -v`
Expected: FAIL (methods don't exist)

**Step 3: Add new methods to ExchangeClient**

Add to `orchestrator/src/orchestrator/exchange/client.py`:

```python
async def fetch_funding_rate_history(self, symbol: str, *, limit: int = 30) -> list[float]:
    """Fetch recent funding rate history via Binance API."""
    rates = await self._exchange.fetch_funding_rate_history(symbol, limit=limit)
    return [r.get("fundingRate", 0.0) for r in rates]

async def fetch_open_interest(self, symbol: str) -> float:
    """Fetch current open interest."""
    result = await self._exchange.fetch_open_interest(symbol)
    return result.get("openInterestAmount", 0.0)

async def fetch_long_short_ratio(self, symbol: str) -> float:
    """Fetch global long/short account ratio."""
    # Binance-specific endpoint via CCXT
    result = await self._exchange.fetch_long_short_ratio_history(symbol, limit=1)
    if result:
        return result[0].get("longShortRatio", 1.0)
    return 1.0

async def fetch_top_trader_long_short_ratio(self, symbol: str) -> float:
    """Fetch top trader long/short ratio."""
    result = await self._exchange.fetch_long_short_ratio_history(
        symbol, limit=1, params={"traderType": "top"}
    )
    if result:
        return result[0].get("longShortRatio", 1.0)
    return 1.0

async def fetch_order_book(self, symbol: str, *, limit: int = 20) -> dict:
    """Fetch order book."""
    return await self._exchange.fetch_order_book(symbol, limit=limit)
```

**Step 4: Add new methods to DataFetcher**

Add to `orchestrator/src/orchestrator/exchange/data_fetcher.py`:

```python
async def fetch_positioning_data(self, symbol: str) -> dict:
    """Fetch all positioning-related data for the Positioning agent."""
    funding_hist, oi, ls_ratio, top_ls_ratio, order_book = await asyncio.gather(
        self._client.fetch_funding_rate_history(symbol),
        self._client.fetch_open_interest(symbol),
        self._client.fetch_long_short_ratio(symbol),
        self._client.fetch_top_trader_long_short_ratio(symbol),
        self._client.fetch_order_book(symbol),
    )

    bid_depth = sum(bid[1] for bid in order_book.get("bids", []))
    ask_depth = sum(ask[1] for ask in order_book.get("asks", []))

    # Calculate OI change % (compare first vs last if history available)
    oi_change_pct = 0.0  # TODO: needs OI history for real calculation

    return {
        "funding_rate_history": funding_hist,
        "open_interest": oi,
        "oi_change_pct": oi_change_pct,
        "long_short_ratio": ls_ratio,
        "top_trader_long_short_ratio": top_ls_ratio,
        "order_book_summary": {"bid_depth": bid_depth, "ask_depth": ask_depth},
    }

async def fetch_macro_indicators(self, symbol: str) -> dict:
    """Fetch weekly candles and calculate 200W MA + Bull Market Support Band."""
    weekly_ohlcv = await self._client.fetch_ohlcv(symbol, "1w", limit=210)

    closes = [candle[4] for candle in weekly_ohlcv]

    # 200W SMA
    ma_200w = sum(closes[-200:]) / min(len(closes), 200) if closes else 0.0

    # Bull Market Support Band: 20W SMA + 21W EMA
    sma_20w = sum(closes[-20:]) / min(len(closes), 20) if closes else 0.0

    # 21W EMA
    ema_21w = _calculate_ema(closes, 21)

    return {
        "ma_200w": ma_200w,
        "bull_support_upper": max(sma_20w, ema_21w),
        "bull_support_lower": min(sma_20w, ema_21w),
    }


def _calculate_ema(values: list[float], period: int) -> float:
    """Calculate EMA for the given period."""
    if not values or period <= 0:
        return 0.0
    if len(values) < period:
        return sum(values) / len(values)

    multiplier = 2 / (period + 1)
    ema = sum(values[:period]) / period

    for value in values[period:]:
        ema = (value - ema) * multiplier + ema

    return ema
```

**Step 5: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_data_fetcher_extended.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/exchange/client.py orchestrator/src/orchestrator/exchange/data_fetcher.py orchestrator/tests/unit/test_data_fetcher_extended.py
git commit -m "feat: extend DataFetcher and ExchangeClient for positioning and macro data"
```

---

## Task 8: Add external data fetchers (Correlation + Catalyst data sources)

**Files:**
- Create: `orchestrator/src/orchestrator/exchange/external_data.py`
- Test: `orchestrator/tests/unit/test_external_data.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_external_data.py
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.exchange.external_data import ExternalDataFetcher


class TestExternalDataFetcher:
    @pytest.mark.asyncio
    async def test_fetch_dxy_data(self):
        fetcher = ExternalDataFetcher()
        with patch("orchestrator.exchange.external_data.aiohttp") as mock_aiohttp:
            mock_response = AsyncMock()
            mock_response.json.return_value = {
                "chart": {"result": [{"indicators": {"quote": [{"close": [104.0, 104.2, 104.5]}]}}]}
            }
            mock_response.status = 200
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_aiohttp.ClientSession.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_aiohttp.ClientSession.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await fetcher.fetch_dxy_data()
            assert "current" in result
            assert "trend_5d" in result

    @pytest.mark.asyncio
    async def test_fetch_dxy_data_fallback_on_error(self):
        fetcher = ExternalDataFetcher()
        with patch("orchestrator.exchange.external_data.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession.side_effect = Exception("Network error")

            result = await fetcher.fetch_dxy_data()
            assert result["current"] == 0.0
            assert result["trend_5d"] == []
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_external_data.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# orchestrator/src/orchestrator/exchange/external_data.py
from __future__ import annotations

import structlog
import aiohttp

logger = structlog.get_logger(__name__)


class ExternalDataFetcher:
    """Fetches cross-market data from free external APIs."""

    async def fetch_dxy_data(self) -> dict:
        """Fetch DXY data from Yahoo Finance."""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?range=5d&interval=1d"
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return self._default_market_data()
                    data = await resp.json()
                    closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
                    closes = [c for c in closes if c is not None]
                    current = closes[-1] if closes else 0.0
                    change_pct = ((closes[-1] / closes[0]) - 1) * 100 if len(closes) >= 2 else 0.0
                    return {"current": current, "change_pct": change_pct, "trend_5d": closes}
        except Exception:
            logger.warning("dxy_fetch_failed")
            return self._default_market_data()

    async def fetch_sp500_data(self) -> dict:
        """Fetch S&P 500 data from Yahoo Finance."""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?range=5d&interval=1d"
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return self._default_market_data()
                    data = await resp.json()
                    closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
                    closes = [c for c in closes if c is not None]
                    current = closes[-1] if closes else 0.0
                    change_pct = ((closes[-1] / closes[0]) - 1) * 100 if len(closes) >= 2 else 0.0
                    return {"current": current, "change_pct": change_pct, "trend_5d": closes}
        except Exception:
            logger.warning("sp500_fetch_failed")
            return self._default_market_data()

    async def fetch_btc_dominance(self) -> dict:
        """Fetch BTC dominance from CoinGecko (free)."""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.coingecko.com/api/v3/global"
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return {"current": 0.0, "change_7d": 0.0}
                    data = await resp.json()
                    btc_dom = data["data"]["market_cap_percentage"].get("btc", 0.0)
                    change = data["data"].get("market_cap_change_percentage_24h_usd", 0.0)
                    return {"current": btc_dom, "change_7d": change}
        except Exception:
            logger.warning("btc_dominance_fetch_failed")
            return {"current": 0.0, "change_7d": 0.0}

    async def fetch_economic_calendar(self) -> list[dict]:
        """Fetch upcoming economic events. Returns empty list as placeholder.

        TODO: integrate with a free economic calendar API.
        """
        return []

    async def fetch_exchange_announcements(self, exchange_id: str = "binance") -> list[str]:
        """Fetch exchange announcements. Returns empty list as placeholder.

        TODO: integrate with Binance announcement API.
        """
        return []

    @staticmethod
    def _default_market_data() -> dict:
        return {"current": 0.0, "change_pct": 0.0, "trend_5d": []}
```

**Step 4: Add aiohttp dependency**

Run: `cd orchestrator && uv add aiohttp`

**Step 5: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_external_data.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/exchange/external_data.py orchestrator/tests/unit/test_external_data.py orchestrator/pyproject.toml orchestrator/uv.lock
git commit -m "feat: add ExternalDataFetcher for DXY, S&P 500, BTC dominance"
```

---

## Task 9: Update PipelineRunner to use new agents

**Files:**
- Modify: `orchestrator/src/orchestrator/pipeline/runner.py`
- Test: `orchestrator/tests/unit/test_pipeline_runner_v2.py`

This is the most critical change. The PipelineRunner needs to:
1. Accept the new agents in its constructor
2. Fetch additional data (positioning, macro, correlation, catalyst)
3. Run all 5 agents in parallel (Technical×2 + Positioning + Catalyst + Correlation)
4. Pass all outputs to Proposer
5. Update PipelineResult to reflect new agent outputs

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_pipeline_runner_v2.py
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.agents.base import AgentResult
from orchestrator.exchange.data_fetcher import DataFetcher, MarketSnapshot
from orchestrator.models import (
    CatalystReport,
    CorrelationAnalysis,
    EntryOrder,
    Momentum,
    PositioningAnalysis,
    Side,
    TechnicalAnalysis,
    TradeProposal,
    Trend,
    VolatilityRegime,
)
from orchestrator.pipeline.runner import PipelineRunner


def make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT:USDT", timeframe="4h",
        current_price=95200.0, volume_24h=1_000_000.0,
        funding_rate=0.0001, ohlcv=[],
    )


def make_technical(label: str = "short_term") -> TechnicalAnalysis:
    return TechnicalAnalysis(
        label=label, trend=Trend.UP, trend_strength=28.0,
        volatility_regime=VolatilityRegime.MEDIUM, volatility_pct=2.3,
        momentum=Momentum.BULLISH, rsi=62.0, key_levels=[], risk_flags=[],
    )


def make_positioning() -> PositioningAnalysis:
    return PositioningAnalysis(
        funding_trend="stable", funding_extreme=False, oi_change_pct=2.0,
        retail_bias="neutral", smart_money_bias="long", squeeze_risk="none",
        liquidity_assessment="normal", risk_flags=[], confidence=0.7,
    )


def make_catalyst() -> CatalystReport:
    return CatalystReport(
        upcoming_events=[], active_events=[],
        risk_level="low", recommendation="proceed", confidence=0.8,
    )


def make_correlation() -> CorrelationAnalysis:
    return CorrelationAnalysis(
        dxy_trend="stable", dxy_impact="neutral",
        sp500_regime="risk_on", btc_dominance_trend="stable",
        cross_market_alignment="favorable", risk_flags=[], confidence=0.7,
    )


def make_proposal() -> TradeProposal:
    return TradeProposal(
        symbol="BTC/USDT:USDT", side=Side.LONG,
        entry=EntryOrder(type="market"),
        position_size_risk_pct=1.0, stop_loss=93000.0,
        take_profit=[], suggested_leverage=10,
        time_horizon="4h", confidence=0.7,
        invalid_if=[], rationale="Test",
    )


class TestPipelineRunnerV2:
    @pytest.mark.asyncio
    async def test_execute_calls_all_agents_in_parallel(self):
        mock_fetcher = AsyncMock(spec=DataFetcher)
        mock_fetcher.fetch_snapshot.return_value = make_snapshot()
        mock_fetcher.fetch_positioning_data.return_value = {
            "funding_rate_history": [], "open_interest": 0, "oi_change_pct": 0,
            "long_short_ratio": 1.0, "top_trader_long_short_ratio": 1.0,
            "order_book_summary": {"bid_depth": 0, "ask_depth": 0},
        }
        mock_fetcher.fetch_macro_indicators.return_value = {
            "ma_200w": 40000.0, "bull_support_upper": 65000.0, "bull_support_lower": 63000.0,
        }

        mock_tech_short = AsyncMock()
        mock_tech_short.analyze.return_value = AgentResult(output=make_technical("short_term"))
        mock_tech_long = AsyncMock()
        mock_tech_long.analyze.return_value = AgentResult(output=make_technical("long_term"))
        mock_positioning = AsyncMock()
        mock_positioning.analyze.return_value = AgentResult(output=make_positioning())
        mock_catalyst = AsyncMock()
        mock_catalyst.analyze.return_value = AgentResult(output=make_catalyst())
        mock_correlation = AsyncMock()
        mock_correlation.analyze.return_value = AgentResult(output=make_correlation())
        mock_proposer = AsyncMock()
        mock_proposer.analyze.return_value = AgentResult(output=make_proposal())

        mock_external = AsyncMock()
        mock_external.fetch_dxy_data.return_value = {"current": 0, "change_pct": 0, "trend_5d": []}
        mock_external.fetch_sp500_data.return_value = {"current": 0, "change_pct": 0, "trend_5d": []}
        mock_external.fetch_btc_dominance.return_value = {"current": 0, "change_7d": 0}
        mock_external.fetch_economic_calendar.return_value = []
        mock_external.fetch_exchange_announcements.return_value = []

        runner = PipelineRunner(
            data_fetcher=mock_fetcher,
            technical_short_agent=mock_tech_short,
            technical_long_agent=mock_tech_long,
            positioning_agent=mock_positioning,
            catalyst_agent=mock_catalyst,
            correlation_agent=mock_correlation,
            proposer_agent=mock_proposer,
            external_data_fetcher=mock_external,
            pipeline_repo=MagicMock(),
            llm_call_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        result = await runner.execute("BTC/USDT:USDT", timeframe="4h")

        # Verify all agents were called
        mock_tech_short.analyze.assert_called_once()
        mock_tech_long.analyze.assert_called_once()
        mock_positioning.analyze.assert_called_once()
        mock_catalyst.analyze.assert_called_once()
        mock_correlation.analyze.assert_called_once()
        mock_proposer.analyze.assert_called_once()
```

**Step 2-6:** Implement the updated PipelineRunner, run tests, commit.

Key changes to `PipelineRunner.__init__`:
- Replace `sentiment_agent` and `market_agent` with `technical_short_agent`, `technical_long_agent`, `positioning_agent`, `catalyst_agent`, `correlation_agent`
- Add `external_data_fetcher` parameter

Key changes to `PipelineRunner.execute()`:
- Step 1: Fetch snapshot + positioning data + macro indicators + external data (all in parallel)
- Step 2: Run 5 analysis agents in parallel via `asyncio.gather`
- Step 3: Run Proposer with all 5 outputs
- Steps 4-7: Same as before (aggregation, risk check, approval/execution, save)

Update `PipelineResult` to replace `sentiment`/`market` fields with new agent output fields.

**Commit:**

```bash
git commit -m "feat: update PipelineRunner for 5-agent parallel pipeline"
```

---

## Task 10: Update Proposer agent + SKILL.md

**Files:**
- Modify: `orchestrator/src/orchestrator/agents/proposer.py`
- Modify: `.claude/skills/proposer/SKILL.md`
- Modify: `orchestrator/tests/unit/test_agent_proposer.py`

The Proposer now receives 5 analysis outputs instead of 2. Update `_build_prompt` to format all sections.

**Key changes to `ProposerAgent._build_prompt()`:**
- Accept kwargs: `snapshot`, `technical_short`, `technical_long`, `positioning`, `catalyst`, `correlation`
- Format 5 sections in the prompt (as specified in design doc)
- Remove old `sentiment` and `market` kwargs

**Key changes to SKILL.md:**
- Update Input Description to list all 5 analysis sections
- Update methodology to incorporate positioning (squeeze risk, crowding), catalyst (event risk), and correlation (cross-market alignment) into the edge assessment
- Add new decision rules:
  - If `catalyst.recommendation == "wait"` → flat
  - If `correlation.cross_market_alignment == "unfavorable"` → reduce position size
  - If `positioning.squeeze_risk != "none"` and direction matches squeeze → reduce leverage
  - If `positioning.liquidity_assessment == "thin"` → reduce position size

**Commit:**

```bash
git commit -m "feat: update Proposer to consume 5 analysis sources"
```

---

## Task 11: Update __main__.py to wire new agents

**Files:**
- Modify: `orchestrator/src/orchestrator/__main__.py`

**Key changes:**
- Import new agent classes: `TechnicalAgent`, `PositioningAgent`, `CatalystAgent`, `CorrelationAgent`
- Import `ExternalDataFetcher`
- Replace `SentimentAgent` and `MarketAgent` instantiation with:
  ```python
  technical_short = TechnicalAgent(client=llm_client, label="short_term", candle_count=50, max_retries=llm_max_retries)
  technical_long = TechnicalAgent(client=llm_client, label="long_term", candle_count=30, max_retries=llm_max_retries)
  positioning_agent = PositioningAgent(client=llm_client, max_retries=llm_max_retries)
  catalyst_agent = CatalystAgent(client=llm_client, max_retries=llm_max_retries)
  correlation_agent = CorrelationAgent(client=llm_client, max_retries=llm_max_retries)
  ```
- Create `ExternalDataFetcher` instance
- Pass new agents to `PipelineRunner`
- Update `EvalRunner` constructor (or defer eval update to a later task)

**Commit:**

```bash
git commit -m "feat: wire new agents into application bootstrap"
```

---

## Task 12: Update EvalRunner for new pipeline

**Files:**
- Modify: `orchestrator/src/orchestrator/eval/runner.py`
- Modify eval datasets if needed

**Key changes:**
- Update `EvalRunner.__init__` to accept new agent types
- Update `_evaluate_case` to run the new 5-agent pipeline
- Add scoring for new agent outputs (or mark as future work)

**Commit:**

```bash
git commit -m "feat: update EvalRunner for expanded agent pipeline"
```

---

## Task 13: Clean up old Sentiment agent

**Files:**
- Delete: `orchestrator/src/orchestrator/agents/sentiment.py`
- Delete: `orchestrator/tests/unit/test_agent_sentiment.py`
- Delete: `.claude/skills/sentiment/SKILL.md`
- Modify: `orchestrator/src/orchestrator/models.py` (mark `SentimentReport` and `KeyEvent` as deprecated or remove if no other code references them)

**Step 1: Search for remaining references**

Run: `cd orchestrator && grep -r "SentimentReport\|SentimentAgent\|sentiment_agent\|sentiment_result" src/ tests/`

**Step 2: Remove or update all references**

**Step 3: Run full test suite**

Run: `cd orchestrator && uv run pytest -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git commit -m "refactor: remove Sentiment agent (absorbed by Technical + Positioning)"
```

---

## Task 14: Integration test — full pipeline

**Files:**
- Create: `orchestrator/tests/integration/test_pipeline_integration.py`

Write an integration test that mocks only the LLM and exchange calls, but exercises the full pipeline flow:
1. DataFetcher returns realistic mock data
2. All 5 agents run and produce valid outputs
3. Proposer synthesizes and produces a proposal
4. Aggregation and risk check work correctly
5. PipelineResult contains all expected fields

**Commit:**

```bash
git commit -m "test: add integration test for expanded 5-agent pipeline"
```

---

## Task 15: Final verification

**Step 1: Run full test suite with coverage**

Run: `cd orchestrator && uv run pytest -v --cov=orchestrator --cov-report=term-missing`
Expected: 80%+ coverage, all tests pass

**Step 2: Run linter**

Run: `cd orchestrator && uv run ruff check src/ tests/`
Expected: No errors

**Step 3: Commit any final fixes**

```bash
git commit -m "chore: final cleanup and coverage verification"
```
