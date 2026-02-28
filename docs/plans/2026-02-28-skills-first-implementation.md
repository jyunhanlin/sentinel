# Skills-First Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate agent prompts from Python classes to `.claude/skills/` files, so Claude reads skill methodology via Read tool, thinks through analysis, and returns structured JSON — improving analysis quality and separating concerns.

**Architecture:** Each agent becomes a thin Python wrapper that formats data and references a skill by name. The skill (a markdown file in `.claude/skills/{name}/SKILL.md`) contains the full analysis methodology, decision criteria, and output schema. `claude -p` reads the skill itself, thinks through the methodology, then outputs a fenced JSON block that Python extracts.

**Tech Stack:** Python 3.12+, Pydantic v2 (frozen models), asyncio, structlog, pytest, uv

---

### Task 1: Add `extract_json_block` utility to schema_validator

This is the foundation — we need a function that extracts JSON from mixed text+JSON responses (since skills instruct Claude to "think first, then output JSON").

**Files:**
- Modify: `orchestrator/src/orchestrator/llm/schema_validator.py:23-38`
- Test: `orchestrator/tests/unit/test_schema_validator.py`

**Step 1: Write failing tests for `extract_json_block`**

Add these tests to `orchestrator/tests/unit/test_schema_validator.py`:

```python
class TestExtractJsonBlock:
    def test_extracts_from_fenced_json_block(self):
        text = 'Some analysis...\n\n```json\n{"score": 42}\n```\n\nMore text.'
        result = _extract_json(text)
        assert result is not None
        assert '"score": 42' in result

    def test_extracts_from_plain_json_object(self):
        text = '{"score": 42}'
        result = _extract_json(text)
        assert result is not None
        assert '"score": 42' in result

    def test_extracts_json_surrounded_by_analysis(self):
        text = (
            "## Analysis\n"
            "The market looks bullish.\n\n"
            "```json\n"
            '{"trend": "up", "confidence": 0.8}\n'
            "```\n\n"
            "This concludes my analysis."
        )
        result = _extract_json(text)
        assert result is not None
        data = json.loads(result)
        assert data["trend"] == "up"

    def test_returns_none_for_no_json(self):
        text = "Just plain text with no JSON anywhere."
        result = _extract_json(text)
        assert result is None
```

**Step 2: Run tests to verify they pass (they should — `_extract_json` already handles code blocks)**

Run: `cd orchestrator && uv run pytest tests/unit/test_schema_validator.py::TestExtractJsonBlock -v`
Expected: PASS — the existing `_extract_json` already handles ```` ```json ``` ```` blocks and bare JSON objects.

> **Note:** The existing `_extract_json` in `schema_validator.py` already supports both fenced code blocks and bare JSON. No code change needed. This task verifies the existing function covers the new use case. If any test fails, extend `_extract_json` accordingly.

**Step 3: Commit**

```bash
cd orchestrator
git add tests/unit/test_schema_validator.py
git commit -m "test: verify _extract_json handles mixed text+JSON for skills migration"
```

---

### Task 2: Create the `sentiment` skill

Write the SKILL.md with full analysis methodology, and examples.md with few-shot demonstrations.

**Files:**
- Create: `.claude/skills/sentiment/SKILL.md`
- Create: `.claude/skills/sentiment/examples.md`

**Step 1: Create `.claude/skills/sentiment/SKILL.md`**

```markdown
---
name: sentiment
description: Crypto market sentiment analysis — feeds into trade proposer
---

# Crypto Sentiment Analyst

## Context

You are part of a crypto futures trading pipeline. Your job is to analyze market data
and produce a sentiment assessment. Your output feeds directly into the **proposer**
skill, which uses your sentiment score and key events to decide whether to trade.

Only output fields the proposer actually consumes. Keep it lean.

## Input Description

| Field | Type | Meaning |
|-------|------|---------|
| symbol | string | Trading pair (e.g. BTC/USDT:USDT) |
| current_price | float | Latest price |
| volume_24h | float | 24-hour trading volume in quote currency |
| funding_rate | float | Perpetual futures funding rate (8h) |
| timeframe | string | Candle timeframe (e.g. 1h) |
| ohlcv | table | Recent OHLCV candles: O, H, L, C, V |

## Methodology

Think through each step before producing output.

### Step 1: Funding Rate Signal
- Positive > 0.01%: crowded longs, potential squeeze risk
- Positive > 0.05%: extreme greed, high reversal probability
- Negative < -0.01%: shorts paying longs, contrarian bullish
- Negative < -0.05%: extreme fear, capitulation possible
- Near zero (±0.005%): neutral, no directional bias from funding

### Step 2: Price Action & Volume
- Rising price + rising volume → strong conviction move
- Rising price + falling volume → weakening momentum, divergence
- Falling price + rising volume → panic selling or distribution
- Falling price + falling volume → selling exhaustion, potential reversal
- Tight range + low volume → consolidation, expect breakout

### Step 3: Candle Structure
- Look at the last 5-10 candles for pattern
- Long wicks on top → selling pressure / rejection
- Long wicks on bottom → buying pressure / absorption
- Consecutive green closes → bullish momentum
- Consecutive red closes → bearish momentum
- Doji/spinning tops → indecision

### Step 4: Synthesize
- Combine signals from steps 1-3
- Weight funding rate heavily for short-term (< 4h) sentiment
- Weight price action more for medium-term (4h-1d) sentiment
- If signals conflict, lean toward neutral (score closer to 50)

## Decision Criteria

| Score Range | Meaning |
|-------------|---------|
| 0-20 | Extreme fear / bearish. Multiple strong bearish signals aligned |
| 20-40 | Bearish. Majority of signals negative |
| 40-60 | Neutral / mixed. Conflicting signals or no clear direction |
| 60-80 | Bullish. Majority of signals positive |
| 80-100 | Extreme greed / bullish. Multiple strong bullish signals aligned |

| Confidence | When to use |
|------------|-------------|
| 0.0-0.3 | Very limited data or highly conflicting signals |
| 0.3-0.5 | Some signals present but ambiguous |
| 0.5-0.7 | Clear signals in one direction with minor conflicts |
| 0.7-0.9 | Strong alignment across multiple signals |
| 0.9-1.0 | Reserved for extreme, unmistakable conditions |

## Output

After your analysis, output a single fenced JSON block:

```json
{
  "sentiment_score": 0-100,
  "key_events": [
    {"event": "description", "impact": "positive|negative|neutral"}
  ],
  "confidence": 0.0-1.0
}
```

Field notes:
- `sentiment_score`: integer 0-100. 50 = neutral. Higher = more bullish.
- `key_events`: notable observations from your analysis. Keep to 1-3 max. Each event should be a concrete observation (e.g. "funding rate at 0.08% signals extreme greed"), not a generic statement.
- `confidence`: how confident you are in the score. See Decision Criteria table above.

Do NOT include a `sources` field — it is not consumed downstream.

## Historical Context

If a "Historical Context" section is provided in the input data, factor past trade outcomes
into your analysis. For example, if recent long trades failed during similar funding conditions,
adjust your sentiment score accordingly.
```

**Step 2: Create `.claude/skills/sentiment/examples.md`**

```markdown
# Sentiment Skill Examples

## Example 1: Bullish with high funding (conflicting signals)

**Input:**
```
Symbol: BTC/USDT:USDT
Current Price: 95200.0
24h Volume: 2,500,000,000
Funding Rate: 0.000800
Timeframe: 1h

Recent OHLCV (5 candles):
  O=94000.0 H=94800.0 L=93800.0 C=94700.0 V=450000
  O=94700.0 H=95100.0 L=94500.0 C=95000.0 V=520000
  O=95000.0 H=95400.0 L=94900.0 C=95300.0 V=480000
  O=95300.0 H=95600.0 L=95100.0 C=95200.0 V=390000
  O=95200.0 H=95500.0 L=95000.0 C=95200.0 V=350000
```

**Reasoning:**
- Funding rate at 0.08% is elevated — crowded longs, squeeze risk
- Price trending up with 5 consecutive green candles — bullish momentum
- Volume declining on last 2 candles — momentum weakening
- Conflict: price bullish but funding overheated and volume fading

**Output:**
```json
{
  "sentiment_score": 58,
  "key_events": [
    {"event": "funding rate at 0.08% signals crowded longs", "impact": "negative"},
    {"event": "5 consecutive green candles with declining volume", "impact": "neutral"}
  ],
  "confidence": 0.5
}
```

## Example 2: Capitulation / extreme fear

**Input:**
```
Symbol: ETH/USDT:USDT
Current Price: 2800.0
24h Volume: 5,000,000,000
Funding Rate: -0.000600
Timeframe: 1h

Recent OHLCV (5 candles):
  O=3050.0 H=3060.0 L=2950.0 C=2960.0 V=800000
  O=2960.0 H=2970.0 L=2880.0 C=2890.0 V=1200000
  O=2890.0 H=2910.0 L=2820.0 C=2830.0 V=1500000
  O=2830.0 H=2840.0 L=2790.0 C=2810.0 V=900000
  O=2810.0 H=2820.0 L=2795.0 C=2800.0 V=600000
```

**Reasoning:**
- Funding rate at -0.06% is deeply negative — shorts dominating, extreme fear
- Price dropped ~8% over 5 candles — strong bearish move
- Volume spiked on candles 2-3 then declined — selling climax may be passing
- Final candle: small body, low volume → selling exhaustion

**Output:**
```json
{
  "sentiment_score": 25,
  "key_events": [
    {"event": "funding rate at -0.06% indicates capitulation", "impact": "negative"},
    {"event": "volume spike then decline suggests selling climax passing", "impact": "positive"}
  ],
  "confidence": 0.7
}
```
```

**Step 3: Commit**

```bash
git add .claude/skills/sentiment/
git commit -m "feat: add sentiment analysis skill with methodology and examples"
```

---

### Task 3: Create the `market` skill

**Files:**
- Create: `.claude/skills/market/SKILL.md`
- Create: `.claude/skills/market/examples.md`

**Step 1: Create `.claude/skills/market/SKILL.md`**

```markdown
---
name: market
description: Crypto technical analysis — feeds into trade proposer
---

# Crypto Technical Analyst

## Context

You are part of a crypto futures trading pipeline. Your job is to analyze OHLCV data,
funding rate, and volume to determine market structure. Your output feeds directly into
the **proposer** skill, which uses your trend, volatility, levels, and risk flags
to generate trade proposals.

## Input Description

| Field | Type | Meaning |
|-------|------|---------|
| symbol | string | Trading pair (e.g. BTC/USDT:USDT) |
| current_price | float | Latest price |
| volume_24h | float | 24-hour trading volume in quote currency |
| funding_rate | float | Perpetual futures funding rate (8h) |
| timeframe | string | Candle timeframe (e.g. 1h) |
| ohlcv | table | Recent OHLCV candles: O, H, L, C, V (up to 20 candles) |

## Methodology

Think through each step before producing output.

### Step 1: Trend Identification
- Compare current price to the first candle's open
- Count green vs red candles
- Look for higher highs + higher lows (uptrend) or lower highs + lower lows (downtrend)
- If price oscillating within a band without clear direction → range

### Step 2: Volatility Assessment
Calculate volatility_pct: for each of the last 14 candles (or all if fewer), compute
True Range = max(high - low, |high - prev_close|, |low - prev_close|).
Average these, divide by current price, multiply by 100.

| volatility_pct | Regime |
|----------------|--------|
| < 1.5% | low |
| 1.5% - 3.5% | medium |
| > 3.5% | high |

### Step 3: Key Levels
Identify support and resistance from the OHLCV data:
- **Support**: price levels where multiple candle lows cluster or where price bounced
- **Resistance**: price levels where multiple candle highs cluster or where price rejected
- Round numbers near current price are psychologically significant
- Only include levels within ±5% of current price
- Maximum 3 support + 3 resistance levels

### Step 4: Risk Flags
Flag conditions that increase trading risk:

| Flag | Trigger |
|------|---------|
| `funding_elevated` | abs(funding_rate) > 0.05% |
| `volume_declining` | last 3 candles volume each lower than previous |
| `high_volatility` | volatility_pct > 5% |
| `near_key_level` | price within 0.3% of support or resistance |
| `trend_exhaustion` | >8 consecutive same-color candles |

## Output

After your analysis, output a single fenced JSON block:

```json
{
  "trend": "up" | "down" | "range",
  "volatility_regime": "low" | "medium" | "high",
  "volatility_pct": <float>,
  "key_levels": [{"type": "support" | "resistance", "price": <float>}],
  "risk_flags": ["<flag_name>"]
}
```

Field notes:
- `trend`: overall market direction from Step 1
- `volatility_regime`: from Step 2 table
- `volatility_pct`: calculated ATR/price percentage from Step 2
- `key_levels`: from Step 3. Empty list if no clear levels
- `risk_flags`: from Step 4. Empty list if no flags triggered

## Historical Context

If a "Historical Context" section is provided in the input data, reference past market
conditions and how they resolved to inform your current analysis.
```

**Step 2: Create `.claude/skills/market/examples.md`**

```markdown
# Market Skill Examples

## Example 1: Uptrend with medium volatility

**Input:**
```
Symbol: BTC/USDT:USDT
Current Price: 95200.0
24h Volume: 2,500,000,000
Funding Rate: 0.000300
Timeframe: 1h

OHLCV Data (10 candles):
  O=93500.0 H=93900.0 L=93300.0 C=93800.0 V=400000
  O=93800.0 H=94200.0 L=93600.0 C=94100.0 V=450000
  O=94100.0 H=94500.0 L=93900.0 C=94400.0 V=420000
  O=94400.0 H=94600.0 L=94200.0 C=94300.0 V=380000
  O=94300.0 H=94700.0 L=94100.0 C=94600.0 V=410000
  O=94600.0 H=95000.0 L=94500.0 C=94900.0 V=440000
  O=94900.0 H=95200.0 L=94700.0 C=95100.0 V=460000
  O=95100.0 H=95400.0 L=94900.0 C=95300.0 V=430000
  O=95300.0 H=95500.0 L=95100.0 C=95200.0 V=400000
  O=95200.0 H=95400.0 L=95000.0 C=95200.0 V=380000
```

**Reasoning:**
- Trend: price moved from 93500 to 95200, series of higher lows → uptrend
- Volatility: average true range ~400 per candle, /95200 * 100 ≈ 0.42% per candle. For 14-candle ATR this extrapolates to ~2.1% → medium
- Support: cluster of lows around 94100-94200 area → support at 94100
- Resistance: highs clustering at 95400-95500 → resistance at 95500
- Risk flags: funding at 0.03% is normal, volume relatively stable, no flags

**Output:**
```json
{
  "trend": "up",
  "volatility_regime": "medium",
  "volatility_pct": 2.1,
  "key_levels": [
    {"type": "support", "price": 94100},
    {"type": "resistance", "price": 95500}
  ],
  "risk_flags": []
}
```

## Example 2: Range-bound with declining volume

**Input:**
```
Symbol: ETH/USDT:USDT
Current Price: 3000.0
24h Volume: 800,000,000
Funding Rate: 0.000050
Timeframe: 1h

OHLCV Data (8 candles):
  O=3010.0 H=3040.0 L=2980.0 C=2990.0 V=200000
  O=2990.0 H=3020.0 L=2975.0 C=3015.0 V=180000
  O=3015.0 H=3035.0 L=2990.0 C=2995.0 V=160000
  O=2995.0 H=3025.0 L=2985.0 C=3010.0 V=150000
  O=3010.0 H=3030.0 L=2980.0 C=2985.0 V=140000
  O=2985.0 H=3015.0 L=2970.0 C=3005.0 V=130000
  O=3005.0 H=3025.0 L=2985.0 C=2995.0 V=120000
  O=2995.0 H=3010.0 L=2980.0 C=3000.0 V=110000
```

**Reasoning:**
- Trend: price oscillating between ~2970 and ~3040, no clear direction → range
- Volatility: true range ~40-50 per candle, /3000 * 100 ≈ 1.5% → low/medium boundary
- Support: multiple lows around 2975-2985 → support at 2975
- Resistance: highs around 3035-3040 → resistance at 3040
- Risk flags: last 3 candles volume declining (130k → 120k → 110k) → volume_declining

**Output:**
```json
{
  "trend": "range",
  "volatility_regime": "low",
  "volatility_pct": 1.4,
  "key_levels": [
    {"type": "support", "price": 2975},
    {"type": "resistance", "price": 3040}
  ],
  "risk_flags": ["volume_declining"]
}
```
```

**Step 3: Commit**

```bash
git add .claude/skills/market/
git commit -m "feat: add market technical analysis skill with methodology and examples"
```

---

### Task 4: Create the `proposer` skill

**Files:**
- Create: `.claude/skills/proposer/SKILL.md`
- Create: `.claude/skills/proposer/examples.md`

**Step 1: Create `.claude/skills/proposer/SKILL.md`**

```markdown
---
name: proposer
description: Crypto futures trade proposal generator — final pipeline output
---

# Trade Proposal Generator

## Context

You are the final stage of a crypto futures trading pipeline. You receive:
1. Raw market data (price, volume, funding rate)
2. Sentiment analysis output (from the sentiment skill)
3. Technical analysis output (from the market skill)

Your job is to synthesize all inputs and decide whether to trade, and if so, generate
a structured trade proposal with entry, stop loss, take profit, and position sizing.

Your output goes to a risk checker, then to trade execution. Be precise with numbers.

## Input Description

| Section | Fields |
|---------|--------|
| Market Data | symbol, current_price, volume_24h, funding_rate |
| Sentiment | sentiment_score (0-100), confidence, key_events |
| Technical | trend (up/down/range), volatility_regime, volatility_pct, key_levels, risk_flags |

## Methodology

Think through each step before producing output.

### Step 1: Edge Assessment
- Is there a clear directional edge?
- Sentiment and trend should agree for a trade
- If sentiment is 40-60 (neutral) AND trend is range → no trade (flat)
- If risk_flags contains > 2 flags → strongly consider flat
- If any agent's confidence is < 0.3 → flat

### Step 2: Direction & Entry
If an edge exists:
- Sentiment > 60 AND trend = up → long
- Sentiment < 40 AND trend = down → short
- Prefer market entry unless price is near a key level where limit makes sense
- For limit entries, set price at nearest support (for long) or resistance (for short)

### Step 3: Stop Loss
- **Long**: stop below nearest support level, or 1-2x ATR below entry
- **Short**: stop above nearest resistance level, or 1-2x ATR above entry
- MUST be: below entry for long, above entry for short
- Minimum distance: 0.5% from entry

### Step 4: Take Profit
- Use 2-3 levels for scaling out
- First TP: 1.5-2x the stop distance (risk-reward ≥ 1.5:1)
- Second TP: 3-4x the stop distance
- Last level MUST have close_pct = 100
- close_pct is % of remaining position, not % of original

### Step 5: Position Sizing (risk %)
- Typical range: 0.5-2.0% of account
- High confidence (>0.7) + low volatility → up to 2.0%
- Medium confidence (0.5-0.7) → 0.5-1.0%
- Low confidence (<0.5) → 0% (flat)
- If risk_flags present, reduce by 0.25% per flag

### Step 6: Leverage
Based on volatility_pct:

| volatility_pct | Max Leverage |
|----------------|-------------|
| < 2% | up to 20x |
| 2-4% | up to 10x |
| > 4% | up to 5x |

Additional constraints:
- If confidence < 0.5 → cap at 5x
- If risk_flags present → reduce by 2x per flag (minimum 1x)
- Round to nearest integer

### Step 7: Invalidation Conditions
List 1-3 concrete conditions that would invalidate this trade:
- Price levels that negate the thesis
- Time-based expiry (e.g. "entry not filled within 2h")
- Market structure changes

## Output

After your analysis, output a single fenced JSON block:

```json
{
  "symbol": "<symbol>",
  "side": "long" | "short" | "flat",
  "entry": {"type": "market"} | {"type": "limit", "price": <float>},
  "position_size_risk_pct": <float 0.0-2.0>,
  "stop_loss": <float | null>,
  "take_profit": [{"price": <float>, "close_pct": <int 1-100>}],
  "suggested_leverage": <int 1-50>,
  "time_horizon": "<e.g. 4h, 1d>",
  "confidence": <float 0.0-1.0>,
  "invalid_if": ["<condition>"],
  "rationale": "<1-2 sentence explanation>"
}
```

Field notes:
- If side = "flat": set position_size_risk_pct = 0, stop_loss = null, take_profit = [], confidence should reflect why no trade
- `rationale`: concise explanation of the decision, referencing specific data points
- `time_horizon`: expected duration of the trade

## Historical Context

If a "Historical Context" section is provided in the input data, factor past trade performance
into your decision. For example, if the last 3 similar setups resulted in losses, increase
your threshold for taking the trade.
```

**Step 2: Create `.claude/skills/proposer/examples.md`**

```markdown
# Proposer Skill Examples

## Example 1: Bullish setup → long trade

**Input:**
```
=== Market Data ===
Symbol: BTC/USDT:USDT
Current Price: 95200.0
24h Volume: 2,500,000,000
Funding Rate: 0.000300

=== Sentiment Analysis ===
Sentiment Score: 68/100
Confidence: 0.7
Key Events: rising volume on breakout above 95000

=== Technical Analysis ===
Trend: up
Volatility: medium (2.1%)
Key Levels: support=94100, resistance=96500
Risk Flags: none
```

**Reasoning:**
- Edge: sentiment bullish (68) + trend up → directional edge exists
- Direction: long
- Entry: market, since price already above key level at 95000
- Stop loss: below support at 94100, that's ~1.2% below entry → ok
- Take profit: stop distance = 95200 - 94100 = 1100
  - TP1 at 95200 + 1650 (1.5x) = 96850, close 50%
  - TP2 at 96500 (resistance), close 100%
  - Actually 96500 < 96850, so use: TP1 at 96500 close 50%, TP2 at 97500 close 100%
- Risk: confidence 0.7 + medium volatility → 1.0%
- Leverage: volatility 2.1% → up to 10x. Confidence 0.7 → ok. No flags → 10x

**Output:**
```json
{
  "symbol": "BTC/USDT:USDT",
  "side": "long",
  "entry": {"type": "market"},
  "position_size_risk_pct": 1.0,
  "stop_loss": 94100,
  "take_profit": [
    {"price": 96500, "close_pct": 50},
    {"price": 97500, "close_pct": 100}
  ],
  "suggested_leverage": 10,
  "time_horizon": "4h",
  "confidence": 0.65,
  "invalid_if": ["price closes below 94000", "funding rate exceeds 0.1%"],
  "rationale": "Bullish sentiment (68) aligns with uptrend. Support at 94100 provides clear stop. Targeting resistance at 96500 for first exit."
}
```

## Example 2: Mixed signals → flat

**Input:**
```
=== Market Data ===
Symbol: ETH/USDT:USDT
Current Price: 3000.0
24h Volume: 800,000,000
Funding Rate: 0.000050

=== Sentiment Analysis ===
Sentiment Score: 52/100
Confidence: 0.4
Key Events: none

=== Technical Analysis ===
Trend: range
Volatility: low (1.4%)
Key Levels: support=2975, resistance=3040
Risk Flags: volume_declining
```

**Reasoning:**
- Edge: sentiment neutral (52), trend range, low confidence (0.4) → no directional edge
- Multiple disqualifiers: neutral sentiment, range-bound, declining volume
- Decision: flat

**Output:**
```json
{
  "symbol": "ETH/USDT:USDT",
  "side": "flat",
  "entry": {"type": "market"},
  "position_size_risk_pct": 0,
  "stop_loss": null,
  "take_profit": [],
  "suggested_leverage": 1,
  "time_horizon": "4h",
  "confidence": 0.3,
  "invalid_if": [],
  "rationale": "Neutral sentiment (52) in range-bound market with declining volume. No clear edge to trade."
}
```
```

**Step 3: Commit**

```bash
git add .claude/skills/proposer/
git commit -m "feat: add trade proposer skill with decision framework and examples"
```

---

### Task 5: Refactor `BaseAgent` to support skill-based prompts

The core change: replace `_build_messages()` with `_build_prompt()` (a single string that tells Claude which skill to use and provides the data). Keep `_build_messages()` as a compatibility bridge so the refactor is incremental.

**Files:**
- Modify: `orchestrator/src/orchestrator/agents/base.py`
- Test: `orchestrator/tests/unit/test_agent_base.py`

**Step 1: Write failing tests for the new skill-based flow**

Add to `orchestrator/tests/unit/test_agent_base.py`:

```python
class SkillFakeAgent(BaseAgent[SentimentReport]):
    """Agent that uses skill-based prompt instead of messages."""

    output_model = SentimentReport
    _skill_name = "sentiment"

    def _build_prompt(self, **kwargs) -> str:
        return f"Use the {self._skill_name} skill.\n\nData: test data"

    def _build_messages(self, **kwargs) -> list[dict]:
        prompt = self._build_prompt(**kwargs)
        return [{"role": "user", "content": prompt}]

    def _get_default_output(self) -> SentimentReport:
        return SentimentReport(
            sentiment_score=50,
            key_events=[],
            sources=[],
            confidence=0.1,
        )


class TestSkillBasedAgent:
    @pytest.mark.asyncio
    async def test_skill_prompt_sent_as_user_message(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                "Analysis: looks bullish\n\n"
                '```json\n{"sentiment_score": 72, "key_events": [],'
                ' "sources": ["data"], "confidence": 0.8}\n```'
            ),
            model="test",
            input_tokens=200,
            output_tokens=100,
            latency_ms=500,
        )

        agent = SkillFakeAgent(client=mock_client)
        result = await agent.analyze()

        assert result.output.sentiment_score == 72
        assert result.degraded is False

        # Verify prompt is sent as single user message (no system message)
        call_args = mock_client.call.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "sentiment" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_skill_response_with_analysis_text_before_json(self):
        """Skill responses include thinking text before the JSON block."""
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                "## Analysis\n\n"
                "The funding rate is slightly positive at 0.03%, indicating mild bullish sentiment.\n"
                "Volume is stable. Price action shows higher lows.\n\n"
                "```json\n"
                '{"sentiment_score": 62, "key_events": '
                '[{"event": "stable funding rate", "impact": "positive", "source": "market"}], '
                '"sources": ["market_data"], "confidence": 0.6}\n'
                "```\n\n"
                "This concludes the analysis."
            ),
            model="test",
            input_tokens=300,
            output_tokens=200,
            latency_ms=800,
        )

        agent = SkillFakeAgent(client=mock_client)
        result = await agent.analyze()

        assert result.output.sentiment_score == 62
        assert result.degraded is False
```

**Step 2: Run tests to verify they fail**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_base.py::TestSkillBasedAgent -v`
Expected: FAIL — `SkillFakeAgent` doesn't exist yet, `_build_prompt` not on `BaseAgent`.

**Step 3: Update `BaseAgent` to support skill-based flow**

Modify `orchestrator/src/orchestrator/agents/base.py`:

```python
from __future__ import annotations

from abc import ABC, abstractmethod

import structlog
from pydantic import BaseModel

from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.llm.schema_validator import ValidationSuccess, validate_llm_output

logger = structlog.get_logger(__name__)


class AgentResult[T: BaseModel](BaseModel):
    output: T
    degraded: bool = False
    llm_calls: list[LLMCallResult] = []
    messages: list[dict[str, str]] = []

    model_config = {"arbitrary_types_allowed": True}


class BaseAgent[T: BaseModel](ABC):
    output_model: type[T]
    _skill_name: str = ""

    def __init__(self, client: LLMClient, max_retries: int = 1) -> None:
        self._client = client
        self._max_retries = max_retries

    async def analyze(self, *, model_override: str | None = None, **kwargs) -> AgentResult[T]:
        messages = self._build_messages(**kwargs)
        llm_calls: list[LLMCallResult] = []

        for attempt in range(1 + self._max_retries):
            call_result = await self._client.call(messages, model=model_override)
            llm_calls.append(call_result)

            validation = validate_llm_output(call_result.content, self.output_model)

            if isinstance(validation, ValidationSuccess):
                logger.info(
                    "agent_success",
                    agent=self.__class__.__name__,
                    attempt=attempt + 1,
                )
                return AgentResult(
                    output=validation.value,
                    degraded=False,
                    llm_calls=llm_calls,
                    messages=messages,
                )

            # Retry with error feedback
            logger.warning(
                "agent_validation_failed",
                agent=self.__class__.__name__,
                attempt=attempt + 1,
                error=validation.error_message,
            )

            if attempt < self._max_retries:
                messages = self._build_retry_messages(messages, validation.error_message)

        # All retries exhausted — degrade
        logger.warning(
            "agent_degraded",
            agent=self.__class__.__name__,
            total_attempts=1 + self._max_retries,
        )
        return AgentResult(
            output=self._get_default_output(),
            degraded=True,
            llm_calls=llm_calls,
            messages=messages,
        )

    def _build_messages(self, **kwargs) -> list[dict[str, str]]:
        """Build messages for LLM call.

        Skill-based agents override _build_prompt() and this method
        wraps the prompt as a single user message (no system message).
        Legacy agents can override this method directly.
        """
        prompt = self._build_prompt(**kwargs)
        return [{"role": "user", "content": prompt}]

    def _build_prompt(self, **kwargs) -> str:
        """Build prompt string for skill-based agents.

        Subclasses should override this to format data and reference a skill.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _build_prompt() or _build_messages()"
        )

    @abstractmethod
    def _get_default_output(self) -> T:
        ...

    def _build_retry_messages(
        self, original_messages: list[dict[str, str]], error: str
    ) -> list[dict[str, str]]:
        return original_messages + [
            {"role": "assistant", "content": "(invalid output)"},
            {
                "role": "user",
                "content": (
                    f"Your previous response failed validation: {error}\n"
                    "Please respond with ONLY a valid JSON object matching the schema."
                ),
            },
        ]
```

Key changes:
- `_build_messages` is no longer abstract — it has a default implementation that calls `_build_prompt()` and wraps it as a single user message.
- `_build_prompt()` is the new extension point for skill-based agents.
- Legacy agents that override `_build_messages()` still work (the method is still called by `analyze()`).

**Step 4: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_base.py -v`
Expected: ALL PASS — both the old `FakeAgent` tests (which override `_build_messages`) and the new `SkillFakeAgent` tests (which override `_build_prompt`).

**Step 5: Commit**

```bash
cd orchestrator
git add src/orchestrator/agents/base.py tests/unit/test_agent_base.py
git commit -m "refactor: add skill-based prompt support to BaseAgent"
```

---

### Task 6: Migrate `SentimentAgent` to skill-based

**Files:**
- Modify: `orchestrator/src/orchestrator/agents/sentiment.py`
- Modify: `orchestrator/tests/unit/test_agent_sentiment.py`

**Step 1: Update tests to verify skill-based behavior**

Rewrite `orchestrator/tests/unit/test_agent_sentiment.py`:

```python
from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.sentiment import SentimentAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import SentimentReport


def make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        current_price=95200.0,
        volume_24h=1_000_000.0,
        funding_rate=0.0001,
        ohlcv=[
            [1700000000000, 94000.0, 95500.0, 93500.0, 95200.0, 1000.0],
            [1700003600000, 95200.0, 96000.0, 95000.0, 95800.0, 800.0],
        ],
    )


class TestSentimentAgent:
    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                "## Analysis\nBullish signals observed.\n\n"
                '```json\n{"sentiment_score": 72, "key_events": '
                '[{"event": "BTC rally", "impact": "positive", "source": "market"}], '
                '"sources": ["market_data"], "confidence": 0.75}\n```'
            ),
            model="test",
            input_tokens=200,
            output_tokens=100,
            latency_ms=1000,
        )

        agent = SentimentAgent(client=mock_client)
        result = await agent.analyze(snapshot=make_snapshot())

        assert isinstance(result.output, SentimentReport)
        assert result.output.sentiment_score == 72
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_prompt_references_skill_and_contains_data(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content='```json\n{"sentiment_score": 50, "key_events": [], "sources": [], "confidence": 0.5}\n```',
            model="test",
            input_tokens=200,
            output_tokens=100,
            latency_ms=500,
        )

        agent = SentimentAgent(client=mock_client)
        await agent.analyze(snapshot=make_snapshot())

        call_args = mock_client.call.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]

        # Should be a single user message (no system message)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

        prompt = messages[0]["content"]
        # Should reference the skill
        assert "sentiment" in prompt.lower()
        assert "skill" in prompt.lower() or "SKILL.md" in prompt
        # Should contain market data
        assert "BTC/USDT:USDT" in prompt
        assert "95200" in prompt

    @pytest.mark.asyncio
    async def test_degrade_returns_neutral(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken",
            model="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        agent = SentimentAgent(client=mock_client, max_retries=0)
        result = await agent.analyze(snapshot=make_snapshot())

        assert result.degraded is True
        assert result.output.sentiment_score == 50
        assert result.output.confidence <= 0.3
```

**Step 2: Run tests to verify they fail**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_sentiment.py -v`
Expected: FAIL — `test_prompt_references_skill_and_contains_data` fails because current agent uses system+user messages.

**Step 3: Rewrite `SentimentAgent`**

Replace `orchestrator/src/orchestrator/agents/sentiment.py`:

```python
from __future__ import annotations

from orchestrator.agents.base import BaseAgent
from orchestrator.agents.utils import summarize_ohlcv
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.models import SentimentReport


class SentimentAgent(BaseAgent[SentimentReport]):
    output_model = SentimentReport
    _skill_name = "sentiment"

    def _build_prompt(self, **kwargs) -> str:
        snapshot: MarketSnapshot = kwargs["snapshot"]
        ohlcv_summary = summarize_ohlcv(snapshot.ohlcv, max_candles=10)

        data = (
            f"Symbol: {snapshot.symbol}\n"
            f"Current Price: {snapshot.current_price}\n"
            f"24h Volume: {snapshot.volume_24h:,.0f}\n"
            f"Funding Rate: {snapshot.funding_rate:.6f}\n"
            f"Timeframe: {snapshot.timeframe}\n\n"
            f"Recent OHLCV ({len(snapshot.ohlcv)} candles):\n{ohlcv_summary}"
        )

        return (
            f"Use the {self._skill_name} skill to analyze the following market data.\n"
            f"Read .claude/skills/{self._skill_name}/SKILL.md for instructions.\n\n"
            f"=== Market Data ===\n{data}"
        )

    def _get_default_output(self) -> SentimentReport:
        return SentimentReport(
            sentiment_score=50,
            key_events=[],
            sources=["degraded"],
            confidence=0.1,
        )
```

**Step 4: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_sentiment.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd orchestrator
git add src/orchestrator/agents/sentiment.py tests/unit/test_agent_sentiment.py
git commit -m "refactor: migrate SentimentAgent to skill-based prompt"
```

---

### Task 7: Migrate `MarketAgent` to skill-based

**Files:**
- Modify: `orchestrator/src/orchestrator/agents/market.py`
- Modify: `orchestrator/tests/unit/test_agent_market.py`

**Step 1: Update tests**

Rewrite `orchestrator/tests/unit/test_agent_market.py`:

```python
from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.market import MarketAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import MarketInterpretation, Trend


def make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        current_price=95200.0,
        volume_24h=1_000_000.0,
        funding_rate=0.0001,
        ohlcv=[
            [1700000000000, 94000.0, 95500.0, 93500.0, 95200.0, 1000.0],
            [1700003600000, 95200.0, 96000.0, 95000.0, 95800.0, 800.0],
        ],
    )


class TestMarketAgent:
    @pytest.mark.asyncio
    async def test_prompt_references_skill_and_contains_data(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                '```json\n{"trend": "up", "volatility_regime": "medium", "volatility_pct": 2.3, '
                '"key_levels": [], "risk_flags": []}\n```'
            ),
            model="test",
            input_tokens=200,
            output_tokens=100,
            latency_ms=1000,
        )

        agent = MarketAgent(client=mock_client)
        await agent.analyze(snapshot=make_snapshot())

        call_args = mock_client.call.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]

        assert len(messages) == 1
        assert messages[0]["role"] == "user"

        prompt = messages[0]["content"]
        assert "market" in prompt.lower()
        assert "skill" in prompt.lower() or "SKILL.md" in prompt
        assert "BTC/USDT:USDT" in prompt

    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                "## Technical Analysis\nUptrend with medium volatility.\n\n"
                '```json\n{"trend": "up", "volatility_regime": "medium", '
                '"key_levels": [{"type": "support", "price": 93000}], '
                '"risk_flags": ["funding_elevated"]}\n```'
            ),
            model="test",
            input_tokens=200,
            output_tokens=100,
            latency_ms=1000,
        )

        agent = MarketAgent(client=mock_client)
        result = await agent.analyze(snapshot=make_snapshot())

        assert isinstance(result.output, MarketInterpretation)
        assert result.output.trend == Trend.UP
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_degrade_returns_neutral(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken",
            model="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        agent = MarketAgent(client=mock_client, max_retries=0)
        result = await agent.analyze(snapshot=make_snapshot())

        assert result.degraded is True
        assert result.output.trend == Trend.RANGE
```

**Step 2: Run tests to verify they fail**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_market.py -v`
Expected: FAIL on `test_prompt_references_skill_and_contains_data`

**Step 3: Rewrite `MarketAgent`**

Replace `orchestrator/src/orchestrator/agents/market.py`:

```python
from __future__ import annotations

from orchestrator.agents.base import BaseAgent
from orchestrator.agents.utils import summarize_ohlcv
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.models import MarketInterpretation, Trend, VolatilityRegime


class MarketAgent(BaseAgent[MarketInterpretation]):
    output_model = MarketInterpretation
    _skill_name = "market"

    def _build_prompt(self, **kwargs) -> str:
        snapshot: MarketSnapshot = kwargs["snapshot"]
        ohlcv_summary = summarize_ohlcv(snapshot.ohlcv, max_candles=20)

        data = (
            f"Symbol: {snapshot.symbol}\n"
            f"Current Price: {snapshot.current_price}\n"
            f"24h Volume: {snapshot.volume_24h:,.0f}\n"
            f"Funding Rate: {snapshot.funding_rate:.6f}\n"
            f"Timeframe: {snapshot.timeframe}\n\n"
            f"OHLCV Data ({len(snapshot.ohlcv)} candles):\n{ohlcv_summary}"
        )

        return (
            f"Use the {self._skill_name} skill to analyze the following market data.\n"
            f"Read .claude/skills/{self._skill_name}/SKILL.md for instructions.\n\n"
            f"=== Market Data ===\n{data}"
        )

    def _get_default_output(self) -> MarketInterpretation:
        return MarketInterpretation(
            trend=Trend.RANGE,
            volatility_regime=VolatilityRegime.MEDIUM,
            volatility_pct=0.0,
            key_levels=[],
            risk_flags=["analysis_degraded"],
        )
```

**Step 4: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_market.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd orchestrator
git add src/orchestrator/agents/market.py tests/unit/test_agent_market.py
git commit -m "refactor: migrate MarketAgent to skill-based prompt"
```

---

### Task 8: Migrate `ProposerAgent` to skill-based

**Files:**
- Modify: `orchestrator/src/orchestrator/agents/proposer.py`
- Modify: `orchestrator/tests/unit/test_agent_proposer.py`

**Step 1: Update tests**

Rewrite `orchestrator/tests/unit/test_agent_proposer.py`:

```python
from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.proposer import ProposerAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import (
    KeyLevel,
    MarketInterpretation,
    SentimentReport,
    Side,
    TradeProposal,
    Trend,
    VolatilityRegime,
)


def make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        current_price=95200.0,
        volume_24h=1_000_000.0,
        funding_rate=0.0001,
        ohlcv=[[1700000000000, 94000.0, 95500.0, 93500.0, 95200.0, 1000.0]],
    )


def make_sentiment() -> SentimentReport:
    return SentimentReport(
        sentiment_score=72,
        key_events=[],
        sources=["market_data"],
        confidence=0.8,
    )


def make_market() -> MarketInterpretation:
    return MarketInterpretation(
        trend=Trend.UP,
        volatility_regime=VolatilityRegime.MEDIUM,
        volatility_pct=2.5,
        key_levels=[KeyLevel(type="support", price=93000.0)],
        risk_flags=[],
    )


FLAT_RESPONSE = (
    "## Analysis\nNo clear edge.\n\n"
    '```json\n{"symbol": "BTC/USDT:USDT", "side": "flat", '
    '"entry": {"type": "market"}, "position_size_risk_pct": 0, '
    '"stop_loss": null, "take_profit": [], '
    '"time_horizon": "4h", "confidence": 0.5, '
    '"invalid_if": [], "rationale": "No signal"}\n```'
)


class TestProposerAgent:
    @pytest.mark.asyncio
    async def test_prompt_references_skill_and_contains_all_inputs(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=FLAT_RESPONSE,
            model="test",
            input_tokens=300,
            output_tokens=150,
            latency_ms=1000,
        )

        agent = ProposerAgent(client=mock_client)
        await agent.analyze(
            snapshot=make_snapshot(),
            sentiment=make_sentiment(),
            market=make_market(),
        )

        call_args = mock_client.call.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]

        assert len(messages) == 1
        assert messages[0]["role"] == "user"

        prompt = messages[0]["content"]
        # References skill
        assert "proposer" in prompt.lower()
        assert "skill" in prompt.lower() or "SKILL.md" in prompt
        # Contains data from all three inputs
        assert "95200" in prompt
        assert "72" in prompt  # sentiment score
        assert "up" in prompt.lower()  # trend
        assert "2.5" in prompt  # volatility_pct

    @pytest.mark.asyncio
    async def test_successful_proposal(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                "## Trade Analysis\nBullish setup.\n\n"
                '```json\n{"symbol": "BTC/USDT:USDT", "side": "long", '
                '"entry": {"type": "market"}, "position_size_risk_pct": 1.5, '
                '"stop_loss": 93000, "take_profit": [{"price": 97000, "close_pct": 100}], '
                '"time_horizon": "4h", "confidence": 0.75, '
                '"invalid_if": [], "rationale": "Bullish momentum"}\n```'
            ),
            model="test",
            input_tokens=300,
            output_tokens=150,
            latency_ms=1500,
        )

        agent = ProposerAgent(client=mock_client)
        result = await agent.analyze(
            snapshot=make_snapshot(),
            sentiment=make_sentiment(),
            market=make_market(),
        )

        assert isinstance(result.output, TradeProposal)
        assert result.output.side == Side.LONG
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_degrade_returns_flat(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken",
            model="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        agent = ProposerAgent(client=mock_client, max_retries=0)
        result = await agent.analyze(
            snapshot=make_snapshot(),
            sentiment=make_sentiment(),
            market=make_market(),
        )

        assert result.degraded is True
        assert result.output.side == Side.FLAT
```

**Step 2: Run tests to verify they fail**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_proposer.py -v`
Expected: FAIL on `test_prompt_references_skill_and_contains_all_inputs`

**Step 3: Rewrite `ProposerAgent`**

Replace `orchestrator/src/orchestrator/agents/proposer.py`:

```python
from __future__ import annotations

from orchestrator.agents.base import BaseAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.models import (
    EntryOrder,
    MarketInterpretation,
    SentimentReport,
    Side,
    TradeProposal,
)


class ProposerAgent(BaseAgent[TradeProposal]):
    output_model = TradeProposal
    _skill_name = "proposer"

    def _build_prompt(self, **kwargs) -> str:
        snapshot: MarketSnapshot = kwargs["snapshot"]
        sentiment: SentimentReport = kwargs["sentiment"]
        market: MarketInterpretation = kwargs["market"]

        key_levels_str = ", ".join(
            f"{kl.type}={kl.price}" for kl in market.key_levels
        ) or "none identified"

        risk_flags_str = ", ".join(market.risk_flags) or "none"

        data = (
            f"=== Market Data ===\n"
            f"Symbol: {snapshot.symbol}\n"
            f"Current Price: {snapshot.current_price}\n"
            f"24h Volume: {snapshot.volume_24h:,.0f}\n"
            f"Funding Rate: {snapshot.funding_rate:.6f}\n\n"
            f"=== Sentiment Analysis ===\n"
            f"Sentiment Score: {sentiment.sentiment_score}/100\n"
            f"Confidence: {sentiment.confidence}\n"
            f"Key Events: {', '.join(e.event for e in sentiment.key_events) or 'none'}\n\n"
            f"=== Technical Analysis ===\n"
            f"Trend: {market.trend}\n"
            f"Volatility: {market.volatility_regime} ({market.volatility_pct:.1f}%)\n"
            f"Key Levels: {key_levels_str}\n"
            f"Risk Flags: {risk_flags_str}"
        )

        return (
            f"Use the {self._skill_name} skill to generate a trade proposal.\n"
            f"Read .claude/skills/{self._skill_name}/SKILL.md for instructions.\n\n"
            f"{data}"
        )

    def _get_default_output(self) -> TradeProposal:
        return TradeProposal(
            symbol="unknown",
            side=Side.FLAT,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=0.0,
            stop_loss=None,
            take_profit=[],
            time_horizon="4h",
            confidence=0.0,
            invalid_if=[],
            rationale="Analysis degraded — no trade signal generated",
        )
```

**Step 4: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_proposer.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd orchestrator
git add src/orchestrator/agents/proposer.py tests/unit/test_agent_proposer.py
git commit -m "refactor: migrate ProposerAgent to skill-based prompt"
```

---

### Task 9: Simplify `ClaudeCLIBackend` for skill-based flow

Since skill-based agents send a single user message (no system prompt), the CLI backend should handle this cleanly. The system prompt extraction still works (returns `None` when no system message exists), so this is mostly verification.

**Files:**
- Modify: `orchestrator/tests/unit/test_backend.py`

**Step 1: Add test for skill-based (no system prompt) flow**

Add to `orchestrator/tests/unit/test_backend.py`:

```python
class TestClaudeCLIBackendSkillFlow:
    @pytest.mark.asyncio
    async def test_complete_without_system_prompt(self):
        """Skill-based agents send a single user message — no --system-prompt flag."""
        backend = ClaudeCLIBackend()

        cli_output = json.dumps({
            "result": '```json\n{"score": 42}\n```',
            "cost_usd": 0.01,
            "duration_ms": 1500,
            "num_turns": 1,
            "is_error": False,
            "session_id": "test-session",
        })

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (cli_output.encode(), b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            result = await backend.complete(
                messages=[
                    {"role": "user", "content": "Use the sentiment skill.\n\nData: ..."},
                ],
                model="sonnet",
                temperature=0.2,
                max_tokens=2000,
            )

        assert result.content == '```json\n{"score": 42}\n```'

        # Verify --system-prompt is NOT in the command
        exec_args = mock_exec.call_args
        cmd_args = exec_args[0]
        assert "--system-prompt" not in cmd_args
```

**Step 2: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_backend.py::TestClaudeCLIBackendSkillFlow -v`
Expected: PASS — the existing backend already handles missing system prompts correctly (the `if system_prompt:` guard on line 145-146).

**Step 3: Commit**

```bash
cd orchestrator
git add tests/unit/test_backend.py
git commit -m "test: verify CLI backend handles skill-based (no system prompt) flow"
```

---

### Task 10: Update CLAUDE.md with skills registration

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add skills registration to CLAUDE.md**

Append to `CLAUDE.md` after the "Conventions" section:

```markdown

## Trading Skills

When running pipeline analysis via `claude -p`, the following skills are available:
- sentiment: `.claude/skills/sentiment/SKILL.md` — market sentiment analysis
- market: `.claude/skills/market/SKILL.md` — technical analysis
- proposer: `.claude/skills/proposer/SKILL.md` — trade proposal generation

Each skill file contains the full analysis methodology, decision criteria, and output schema.
Agents reference skills by name in their prompts; Claude reads the SKILL.md via the Read tool.
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: register trading skills in CLAUDE.md"
```

---

### Task 11: Run full test suite and verify

**Files:** (no changes, verification only)

**Step 1: Run all tests**

Run: `cd orchestrator && uv run pytest -v --cov=orchestrator`
Expected: ALL PASS with coverage ≥ 80%

**Step 2: Run linter**

Run: `cd orchestrator && uv run ruff check src/ tests/`
Expected: No errors

**Step 3: Fix any failures**

If any tests fail, fix the issue and re-run. Common issues:
- Tests in `test_agent_base.py` that import `FakeAgent` with the old `_build_messages` signature
- Import errors from removed dependencies

**Step 4: Final commit if fixes were needed**

```bash
cd orchestrator
git add -A
git commit -m "fix: resolve test failures from skills migration"
```
