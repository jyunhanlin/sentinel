<!-- Generated: 2026-03-02 | Files scanned: 12 | Token estimate: ~800 -->
# Agents & Skills Codemap

## Overview

Agents are LLM wrappers that call external "skills" (`.claude/skills/`) instead of embedding prompts. This separates domain knowledge (how to analyze) from orchestration logic (what data to pass).

## Architecture

```
BaseAgent[T] (Generic async wrapper, _label for instance identity)
    │
    ├─ TechnicalAgent ×2
    │   ├─ Skill: .claude/skills/technical/SKILL.md
    │   ├─ Instances: short_term (50 candles), long_term (30 candles + macro)
    │   └─ Output: TechnicalAnalysis
    │
    ├─ PositioningAgent
    │   ├─ Skill: .claude/skills/positioning/SKILL.md
    │   ├─ Input: funding history, OI, L/S ratios, order book
    │   └─ Output: PositioningAnalysis
    │
    ├─ CatalystAgent
    │   ├─ Skill: .claude/skills/catalyst/SKILL.md
    │   ├─ Input: economic calendar, exchange announcements
    │   └─ Output: CatalystReport
    │
    ├─ CorrelationAgent
    │   ├─ Skill: .claude/skills/correlation/SKILL.md
    │   ├─ Input: DXY, S&P 500, BTC dominance
    │   └─ Output: CorrelationAnalysis
    │
    └─ ProposerAgent
        ├─ Skill: .claude/skills/proposer/SKILL.md
        ├─ Input: snapshot + all 5 analysis outputs
        └─ Output: TradeProposal
```

## Agent Base Class

**File:** `agents/base.py`

```
BaseAgent[T]:
  _skill_name: str        # skill directory name
  _label: str             # instance identity (e.g. "short_term")
  _log_id → {agent, label}  # common log fields

  analyze(**kwargs) → AgentResult[T]:
    log agent_start
    _build_messages(**kwargs) → messages
    for attempt in range(1 + max_retries):
      client.call(messages) → LLMCallResult
      validate_llm_output(content, output_model)
      if valid → log agent_success, return AgentResult(degraded=False)
      else → log agent_validation_failed, retry with error feedback
    log agent_degraded → return AgentResult(output=_get_default_output(), degraded=True)
```

## Agent Implementations

### TechnicalAgent — `agents/technical.py`
- Constructor: `label` (short_term/long_term), `candle_count` (50/30)
- Prompt: OHLCV summary + macro indicators (long_term only: 200W MA, bull support band)
- Output: `TechnicalAnalysis` (trend, trend_strength, volatility_regime, momentum, rsi, key_levels, risk_flags)
- Default: trend=RANGE, momentum=NEUTRAL, rsi=50

### PositioningAgent — `agents/positioning.py`
- Input: funding_rate_history, open_interest, oi_change_pct, long_short_ratio, top_trader_long_short_ratio, order_book
- Output: `PositioningAnalysis` (funding_trend, funding_extreme, oi_change_pct, retail_bias, smart_money_bias, squeeze_risk, liquidity_assessment)
- Default: all neutral, confidence=0.1

### CatalystAgent — `agents/catalyst.py`
- Input: economic_calendar, exchange_announcements
- Output: `CatalystReport` (upcoming_events, active_events, risk_level, recommendation, confidence)
- Default: no events, risk_level=low

### CorrelationAgent — `agents/correlation.py`
- Input: dxy_data, sp500_data, btc_dominance
- Output: `CorrelationAnalysis` (dxy_trend, dxy_impact, sp500_regime, btc_dominance_trend, cross_market_alignment)
- Default: all neutral/stable, confidence=0.1

### ProposerAgent — `agents/proposer.py`
- Input: snapshot + TechnicalAnalysis(short) + TechnicalAnalysis(long) + PositioningAnalysis + CatalystReport + CorrelationAnalysis
- Output: `TradeProposal` (symbol, side, entry, stop_loss, take_profit, position_size_risk_pct, suggested_leverage, confidence, rationale)
- Default: side=FLAT, confidence=0.0

## Skill Directory

```
.claude/skills/
├── technical/SKILL.md     # Trend, volatility, momentum methodology
├── positioning/SKILL.md   # Funding, OI, squeeze detection
├── catalyst/SKILL.md      # Event impact assessment
├── correlation/SKILL.md   # Cross-market relationship analysis
└── proposer/
    ├── SKILL.md            # Trade decision framework
    └── examples.md         # Example proposals
```

## Logging Events

All events include `agent=ClassName` + `label=short_term` (if set):
- `agent_start` — before `_build_messages`
- `agent_success` — validation passed, includes `attempt`
- `agent_validation_failed` — retry with error feedback
- `agent_degraded` — all retries exhausted, using default output

## Testing

- `tests/unit/test_agent_base.py` — BaseAgent retry/degradation logic
- `tests/unit/test_agent_technical.py` — TechnicalAgent prompt building
- `tests/unit/test_agent_positioning.py` — PositioningAgent
- `tests/unit/test_agent_catalyst.py` — CatalystAgent
- `tests/unit/test_agent_correlation.py` — CorrelationAgent
- `tests/unit/test_agent_proposer.py` — ProposerAgent prompt building
