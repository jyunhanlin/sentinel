# Agents & Skills Codemap

**Last Updated:** 2026-02-28

## Overview

Agents are LLM wrappers that call external "skills" (Markdown files in `.claude/skills/`) instead of embedding prompts. This separates domain knowledge (how to analyze) from orchestration logic (what data to pass).

## Architecture

```
BaseAgent[T] (Generic async wrapper)
    │
    ├─ SentimentAgent
    │   ├─ Skill: .claude/skills/sentiment/SKILL.md
    │   ├─ Input: MarketSnapshot
    │   └─ Output: SentimentReport
    │
    ├─ MarketAgent
    │   ├─ Skill: .claude/skills/market/SKILL.md
    │   ├─ Input: MarketSnapshot
    │   └─ Output: MarketInterpretation
    │
    └─ ProposerAgent
        ├─ Skill: .claude/skills/proposer/SKILL.md
        ├─ Input: MarketSnapshot + SentimentReport + MarketInterpretation
        └─ Output: TradeProposal
```

## Agent Base Class

**File:** `orchestrator/src/orchestrator/agents/base.py`

```python
class BaseAgent[T: BaseModel](ABC):
    def __init__(self, client: LLMClient, max_retries: int = 1) -> None:
        self._client = client
        self._max_retries = max_retries
        self.output_model: type[T]  # Must be defined by subclass
        self._skill_name: str = ""
```

### Public Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `analyze(**kwargs)` | Async entry: calls LLM, validates output, retries on failure | `AgentResult[T]` |

### Protected Methods (Override in subclass)

| Method | Purpose | Default |
|--------|---------|---------|
| `_build_messages(**kwargs)` | Formats prompt with skill reference | Raises NotImplementedError |
| `_build_retry_messages(messages, error)` | Rebuilds messages with validation error feedback | Appends error as user message |
| `_get_default_output()` | Returns fallback output if all retries exhausted | Raises NotImplementedError |

### Execution Flow

```
analyze(**kwargs)
    │
    ├─ _build_messages(**kwargs) → list[dict]
    │
    └─ For attempt in range(1 + max_retries):
        │
        ├─ client.call(messages) → LLMCallResult
        │
        ├─ validate_llm_output(content, output_model)
        │   └─ Extract JSON from fenced code block
        │   └─ Parse + validate against Pydantic model
        │
        ├─ If validation succeeds:
        │   └─ Return AgentResult(output=value, degraded=False)
        │
        └─ If validation fails and attempt < max_retries:
            └─ messages = _build_retry_messages(messages, error)
            └─ Loop again

        If all retries fail:
            └─ Return AgentResult(output=_get_default_output(), degraded=True)
```

### AgentResult[T]

```python
class AgentResult[T: BaseModel](BaseModel):
    output: T  # The analyzed output
    degraded: bool = False  # True if agent used fallback
    llm_calls: list[LLMCallResult] = []  # LLM request/response + timing
    messages: list[dict[str, str]] = []  # Sent messages (for debugging)
```

## Skill-Based Prompts

### Why Skills?

**Before (embedded prompt):**
```python
prompt = f"""You are a sentiment analyst. Analyze:
Symbol: {symbol}
Price: {price}
...
Respond with ONLY JSON."""
```

**After (skill reference):**
```python
prompt = f"""Use the sentiment skill.

=== Market Data ===
Symbol: {symbol}
Price: {price}
..."""
```

Benefits:
- Claude reads `.claude/skills/sentiment/SKILL.md` automatically
- Skill includes methodology, decision criteria, examples
- Allows chain-of-thought thinking before JSON output
- Easier to iterate on analysis without code changes

### Skill Directory Structure

```
.claude/skills/
├── sentiment/
│   ├── SKILL.md          # Role, methodology, output schema
│   └── examples.md       # 2-3 real market examples
├── market/
│   ├── SKILL.md
│   └── examples.md
└── proposer/
    ├── SKILL.md
    └── examples.md
```

### SKILL.md Template

Each skill file has:
1. **YAML frontmatter** — name, description
2. **Context** — Role in pipeline, what output feeds into
3. **Input Description** — Data format explanation
4. **Methodology** — Step-by-step analysis process
5. **Decision Criteria** — When/how to make decisions
6. **Output** — Exact JSON schema with field notes
7. **Historical Context** — (Optional) RAG injection point for past results

Example structure (sentiment):
```markdown
---
name: sentiment
description: Crypto market sentiment analysis — feeds into trade proposer
---

# Crypto Sentiment Analyst

## Context
[Role explanation]

## Input Description
| Field | Type | Meaning |
| ... |

## Methodology
### Step 1: Funding Rate Signal
### Step 2: Price Action & Volume
...

## Decision Criteria
| Score Range | Meaning |

## Output
```json
{
  "sentiment_score": 0-100,
  "key_events": [...],
  "confidence": 0.0-1.0
}
```

[Field notes]
```

## Sentiment Agent

**File:** `orchestrator/src/orchestrator/agents/sentiment.py`

```python
class SentimentAgent(BaseAgent[SentimentReport]):
    output_model = SentimentReport
    _skill_name = "sentiment"
```

### Message Format

```
Use the sentiment skill.

=== Market Data ===
Symbol: BTC/USDT:USDT
Current Price: 45000.50
24h Volume: 28,500,000,000
Funding Rate: 0.000865
Timeframe: 1h

Recent OHLCV (10 candles):
[timestamp] O: 44750, H: 45100, L: 44600, C: 45000, V: 1,234,567
...
```

### Output Model

```python
class SentimentReport(BaseModel, frozen=True):
    sentiment_score: int = Field(ge=0, le=100)  # 50 = neutral
    key_events: list[KeyEvent] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

class KeyEvent(BaseModel, frozen=True):
    event: str  # e.g., "funding rate at 0.08% signals extreme greed"
    impact: str  # "positive", "negative", "neutral"
    source: str = ""
```

### Fallback Output

```python
def _get_default_output(self) -> SentimentReport:
    return SentimentReport(
        sentiment_score=50,  # Neutral
        key_events=[],
        sources=["degraded"],
        confidence=0.1,  # Very low confidence signals degradation
    )
```

## Market Agent

**File:** `orchestrator/src/orchestrator/agents/market.py`

```python
class MarketAgent(BaseAgent[MarketInterpretation]):
    output_model = MarketInterpretation
    _skill_name = "market"
```

### Message Format

Similar to sentiment, includes OHLCV summary and technical indicators.

### Output Model

```python
class MarketInterpretation(BaseModel, frozen=True):
    trend: Trend  # "up", "down", "range"
    volatility_regime: VolatilityRegime  # "low", "medium", "high"
    volatility_pct: float = Field(ge=0.0, default=0.0)
    key_levels: list[KeyLevel]  # Support/resistance prices
    risk_flags: list[str]  # e.g., ["potential_liquidation", "volume_divergence"]

class KeyLevel(BaseModel, frozen=True):
    type: str  # "support", "resistance"
    price: float

class Trend(StrEnum):
    UP = "up"
    DOWN = "down"
    RANGE = "range"

class VolatilityRegime(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
```

### Fallback Output

```python
def _get_default_output(self) -> MarketInterpretation:
    return MarketInterpretation(
        trend=Trend.RANGE,
        volatility_regime=VolatilityRegime.MEDIUM,
        volatility_pct=0.0,
        key_levels=[],
        risk_flags=["degraded"],
    )
```

## Proposer Agent

**File:** `orchestrator/src/orchestrator/agents/proposer.py`

The final stage: consumes sentiment + market, produces a trade proposal.

```python
class ProposerAgent(BaseAgent[TradeProposal]):
    output_model = TradeProposal
    _skill_name = "proposer"
```

### Message Format

```
Use the proposer skill.

=== Market Data ===
Symbol: BTC/USDT:USDT
Current Price: 45000.50
...

=== Sentiment Analysis ===
Score: 72 (bullish)
Confidence: 0.85
Key Events:
  - Funding rate at 0.08% signals extreme greed (negative impact)
  - ...

=== Market Structure ===
Trend: up
Volatility: medium (2.3%)
Key Levels:
  - Support: 44500, 43800
  - Resistance: 46000, 47500
Risk Flags: [volume_divergence]
```

### Output Model

```python
class TradeProposal(BaseModel, frozen=True):
    proposal_id: str  # UUID
    symbol: str
    side: Side  # "long", "short", "flat"
    entry: EntryOrder  # {"type": "market"} or {"type": "limit", "price": float}
    position_size_risk_pct: float  # 0.0-2.0% of account
    stop_loss: float | None
    take_profit: list[TakeProfit]  # [{"price": float, "close_pct": int}]
    suggested_leverage: int  # 1-50x
    time_horizon: str  # "4h", "1d", etc.
    confidence: float  # 0.0-1.0
    invalid_if: list[str]  # Invalidation conditions
    rationale: str  # Why this trade

class EntryOrder(BaseModel, frozen=True):
    type: str  # "market" or "limit"
    price: float | None  # Only set for limit orders

class TakeProfit(BaseModel, frozen=True):
    price: float
    close_pct: int = Field(ge=1, le=100)  # Percent of position to close
```

### Fallback Output

```python
def _get_default_output(self) -> TradeProposal:
    return TradeProposal(
        symbol="unknown",
        side=Side.FLAT,
        entry=EntryOrder(type="market"),
        position_size_risk_pct=0.0,
        stop_loss=None,
        take_profit=[],
        suggested_leverage=1,
        time_horizon="1h",
        confidence=0.1,
        invalid_if=["degraded"],
        rationale="Agent degraded, flat position.",
    )
```

## Agent Utilities

**File:** `orchestrator/src/orchestrator/agents/utils.py`

| Function | Purpose |
|----------|---------|
| `summarize_ohlcv(ohlcv, max_candles)` | Formats OHLCV for human reading |

Example output:
```
[2026-02-28 12:00] O: 44750, H: 45100, L: 44600, C: 45000, V: 1234567
```

## LLM Client

**File:** `orchestrator/src/orchestrator/llm/client.py`

Agents use `LLMClient` to call LLM:

```python
class LLMClient:
    async def call(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> LLMCallResult:
        """Calls LLM backend (API or CLI), returns result with timing/cost."""
```

Returns:
```python
class LLMCallResult(BaseModel):
    content: str  # Response text (may include JSON block)
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
```

## Output Validation

**File:** `orchestrator/src/orchestrator/llm/schema_validator.py`

Extracts and validates JSON from LLM response:

```python
def validate_llm_output(content: str, model: type[T]) -> ValidationSuccess[T] | ValidationError:
    """
    1. Find JSON block (```json...```)
    2. Parse JSON
    3. Validate against Pydantic model
    4. Return success with parsed object or error with message
    """
```

If LLM embeds JSON in explanatory text:
```
The sentiment score is 72 because:
- Funding rate at 0.08% signals extreme greed
- Price action shows strong momentum

```json
{
  "sentiment_score": 72,
  "key_events": [...],
  "confidence": 0.85
}
```

This extracts and validates the JSON block correctly.

## Testing

Unit tests for agents:

**File:** `tests/unit/test_agent_sentiment.py`
- Mocks LLMClient
- Tests message building
- Tests degradation on validation failure

**File:** `tests/unit/test_agent_proposer.py`
- Tests trade proposal generation
- Validates output against model schema

**File:** `tests/unit/test_eval_consistency.py`
- Tests consistency scoring across agent outputs

## Integration with Pipeline

See [pipeline.md](pipeline.md) for how agents integrate:

1. Pipeline calls `agent.analyze(snapshot=...)`
2. Agent returns `AgentResult` with output + degradation flag
3. Pipeline checks `result.degraded` to trigger fallback logic
4. Aggregator validates multi-agent consistency

## Configuration

Environment variables:
- `LLM_MODEL` — Default model (e.g., "anthropic/claude-sonnet-4-6")
- `LLM_MODEL_PREMIUM` — Override model (e.g., "anthropic/claude-opus-4-6")
- `LLM_TEMPERATURE` — Temperature for all agents (default: 0.2)
- `LLM_MAX_TOKENS` — Max response length (default: 2000)
- `LLM_MAX_RETRIES` — Validation retries (default: 1)

## Related

- [pipeline.md](pipeline.md) — How agents fit in the pipeline
- [evaluation.md](evaluation.md) — Testing agents against golden dataset
- [configuration.md](configuration.md) — LLM settings
