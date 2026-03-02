# Agent Expansion Design

Date: 2026-03-02

## Overview

Expand the Sentinel pipeline from 3 agents (Sentiment, Market, Proposer) to 5 analysis agents + Proposer, optimized for leveraged futures entry decisions.

## Changes Summary

### Remove

- **Sentiment agent** — responsibilities absorbed by Technical (K-line patterns, volume divergence) and Positioning (funding rate)

### Rename + Upgrade

- **Market → Technical** — same SKILL.md, instantiated twice with different parameters:
  - `Technical(timeframe="4h", candle_count=50, label="short_term")`
  - `Technical(timeframe="1d", candle_count=30, label="long_term")`

### New Agents

- **Positioning** — order flow and positioning analysis
- **Catalyst** — news events and economic calendar
- **Correlation** — cross-market relationship analysis

### Unchanged

- **Proposer** — input expanded from 2 to 4 sources (Technical×2, Positioning, Catalyst, Correlation)

## Pipeline

```
DataFetcher
  ├─ OHLCV(4h, 50)  ──────────→ Technical(short_term) ─┐
  ├─ OHLCV(1d, 30)  ──────────→ Technical(long_term)  ──┤
  ├─ OHLCV(1w, 210) ──────────→ calc 200W MA + BMSS   ↗  │
  ├─ OI/Funding/Liq/OrderBook ─→ Positioning ───────────┼→ Proposer → TradeProposal
  ├─ Economic calendar + Exch. ─→ Catalyst ─────────────┤
  └─ DXY/SPX/BTC.D  ──────────→ Correlation ───────────┘
```

All analysis agents run in parallel via `asyncio.gather`, results converge at Proposer.

## Agent Details

### Technical (upgraded from Market)

**Skill:** Single `technical` SKILL.md, parameterized by timeframe.

**Indicators (shared by short and long term):**

| Indicator | Parameters | Purpose |
|-----------|-----------|---------|
| ADX | 14 | Trend strength — ADX < 20 means ranging, dangerous for leverage |
| RSI | 14 | Overbought/oversold — reversal risk |
| MACD | 12, 26, 9 | Momentum direction + divergence |
| Bollinger Bands | 20, 2 | Volatility + overextension |
| EMA | 20, 50 | Trend direction confirmation |
| ATR | 14 | Volatility measurement (existing) |
| Support/Resistance | ±5% range | Key levels (existing) |
| K-line patterns | — | Doji, long wicks, etc. (moved from Sentiment) |
| Volume divergence | — | Price/volume mismatch (moved from Sentiment) |

**Long-term only (additional inputs, pre-calculated by DataFetcher):**

| Indicator | Definition | Purpose |
|-----------|-----------|---------|
| 200W MA | 200-week SMA | Macro bull/bear boundary |
| Bull Market Support Band | 20W SMA + 21W EMA | Bull market pullback buy zone |

**Output model:** `TechnicalAnalysis` (replaces `MarketInterpretation`)

```python
class TechnicalAnalysis(BaseModel, frozen=True):
    label: str                          # "short_term" or "long_term"
    trend: Trend                        # "up" | "down" | "range"
    trend_strength: float               # ADX value
    volatility_regime: VolatilityRegime # "low" | "medium" | "high"
    volatility_pct: float               # ATR/price %
    momentum: Momentum                  # "bullish" | "bearish" | "neutral"
    rsi: float                          # 0-100
    key_levels: list[KeyLevel]
    risk_flags: list[str]
    # Long-term only
    above_200w_ma: bool | None          # null for short-term
    bull_support_band_status: str | None # "above" | "within" | "below" | null
```

### Positioning (new)

**Data sources (all from Binance API):**

| Data | API | Purpose |
|------|-----|---------|
| Funding rate history | `GET /fapi/v1/fundingRate` | Sentiment trend, not just current value |
| Open Interest | `GET /fapi/v1/openInterest` + history | Capital flow, confirm breakout authenticity |
| Long/Short ratio | `GET /futures/data/globalLongShortAccountRatio` | Retail positioning |
| Top trader L/S ratio | `GET /futures/data/topLongShortPositionRatio` | Smart money direction |
| Liquidation data | `GET /fapi/v1/allForceOrders` | Squeeze potential |
| Order book depth | CCXT `fetch_order_book()` | Liquidity assessment (merged from Liquidity) |

**Output model:**

```python
class PositioningAnalysis(BaseModel, frozen=True):
    funding_trend: str                # "rising" | "falling" | "stable"
    funding_extreme: bool             # funding rate at extreme levels
    oi_change_pct: float              # OI change % over period
    retail_bias: str                  # "long" | "short" | "neutral"
    smart_money_bias: str             # "long" | "short" | "neutral"
    squeeze_risk: str                 # "long_squeeze" | "short_squeeze" | "none"
    liquidity_assessment: str         # "thin" | "normal" | "deep"
    risk_flags: list[str]
    confidence: float                 # 0.0-1.0
```

### Catalyst (new)

**Data sources (all free):**

| Data | Source | Purpose |
|------|--------|---------|
| Economic calendar | Free API (e.g., Trading Economics free tier, or scraped) | FOMC, CPI, NFP — high-impact macro events |
| Exchange announcements | Binance API announcements | Listing/delisting, maintenance, policy changes |

**Output model:**

```python
class CatalystReport(BaseModel, frozen=True):
    upcoming_events: list[CatalystEvent]  # next 24-48h
    active_events: list[CatalystEvent]    # currently impacting
    risk_level: str                       # "low" | "medium" | "high"
    recommendation: str                   # "proceed" | "reduce_size" | "wait"
    confidence: float                     # 0.0-1.0

class CatalystEvent(BaseModel, frozen=True):
    event: str                  # e.g. "FOMC Rate Decision"
    time: str                   # ISO timestamp or "ongoing"
    impact: str                 # "high" | "medium" | "low"
    direction_bias: str         # "bullish" | "bearish" | "uncertain"
```

### Correlation (new)

**Data sources (all free):**

| Data | Source | Purpose |
|------|--------|---------|
| DXY (Dollar Index) | Free forex API / Yahoo Finance | USD strength inversely correlated with BTC |
| S&P 500 | Yahoo Finance | Risk-on/risk-off regime |
| BTC Dominance | CCXT or CoinGecko (free) | Capital rotation between BTC and alts |

**Output model:**

```python
class CorrelationAnalysis(BaseModel, frozen=True):
    dxy_trend: str              # "strengthening" | "weakening" | "stable"
    dxy_impact: str             # "headwind" | "tailwind" | "neutral"
    sp500_regime: str           # "risk_on" | "risk_off" | "neutral"
    btc_dominance_trend: str    # "rising" | "falling" | "stable"
    cross_market_alignment: str # "favorable" | "unfavorable" | "mixed"
    risk_flags: list[str]
    confidence: float           # 0.0-1.0
```

### Proposer (updated input)

Proposer prompt now receives 4 analysis sections:

```
=== Short-Term Technical ===
Trend: {short.trend} (strength: {short.trend_strength})
Momentum: {short.momentum} (RSI: {short.rsi})
Volatility: {short.volatility_regime} ({short.volatility_pct:.1f}%)
Key Levels: ...
Risk Flags: ...

=== Long-Term Technical ===
Trend: {long.trend} (strength: {long.trend_strength})
Momentum: {long.momentum} (RSI: {long.rsi})
200W MA: {"above" if long.above_200w_ma else "below"}
Bull Support Band: {long.bull_support_band_status}
Risk Flags: ...

=== Positioning ===
Funding Trend: {pos.funding_trend} (extreme: {pos.funding_extreme})
OI Change: {pos.oi_change_pct:+.1f}%
Retail Bias: {pos.retail_bias}
Smart Money Bias: {pos.smart_money_bias}
Squeeze Risk: {pos.squeeze_risk}
Liquidity: {pos.liquidity_assessment}

=== Catalyst ===
Risk Level: {cat.risk_level}
Recommendation: {cat.recommendation}
Upcoming: {events...}

=== Correlation ===
DXY: {cor.dxy_trend} ({cor.dxy_impact})
S&P 500: {cor.sp500_regime}
BTC Dominance: {cor.btc_dominance_trend}
Alignment: {cor.cross_market_alignment}
```

## Data Sources Summary

| Source | Cost | Agents Using It |
|--------|------|----------------|
| CCXT (Binance) | Free | Technical, Positioning |
| Binance Futures API | Free | Positioning |
| Binance Announcements | Free | Catalyst |
| Economic Calendar API | Free | Catalyst |
| Yahoo Finance / Free Forex API | Free | Correlation |
| CoinGecko (BTC.D) | Free | Correlation |

All data sources are free.

## Trade-offs

| Decision | Gain | Cost |
|----------|------|------|
| Remove Sentiment agent | Cleaner separation of concerns | Lose standalone sentiment score (absorbed into other agents) |
| Single Technical SKILL.md | DRY, easier maintenance | Prompt must handle conditional sections (200W MA only for long-term) |
| Merge Liquidity into Positioning | Fewer agent calls | Positioning prompt gets larger |
| Skip Volatility Structure (Deribit) | No new API integration | Lose options-implied vol signal |
| Skip on-chain analytics | No paid API needed | Lose whale wallet tracking |

## Future Extensions

- **On-chain analytics agent** — when paid API (Glassnode/Nansen) is available
- **Volatility Structure** — Deribit options data for implied vol and put/call ratio
- **Smart Money (standalone)** — when on-chain data enables whale tracking
