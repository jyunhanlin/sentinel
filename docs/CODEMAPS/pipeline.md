# Pipeline Codemap

**Last Updated:** 2026-02-28

## Overview

The pipeline is the core execution engine: fetches market data, runs three LLM agents in parallel, aggregates results, checks risk, and awaits approval.

## Architecture

```
PipelineRunner.execute(symbol, timeframe)
    │
    ├─ DataFetcher.fetch_snapshot(symbol)
    │   └─ CCXT client → OHLCV, current price, funding rate, volume
    │
    ├─ Run 3 agents in parallel:
    │   ├─ SentimentAgent.analyze(snapshot)
    │   ├─ MarketAgent.analyze(snapshot)
    │   └─ ProposerAgent.analyze(snapshot, sentiment, market)
    │
    ├─ Aggregator.aggregate_proposal(proposal, sentiment, market)
    │   └─ Combines outputs with confidence checks
    │
    ├─ RiskChecker.check(proposal, open_exposure, daily_loss)
    │   └─ Returns approval or pause/reject
    │
    ├─ ApprovalManager.create_approval(proposal)
    │   └─ Sends Telegram message, waits for button click
    │
    └─ OrderExecutor.execute_entry(proposal)
        └─ Paper or live execution, returns ExecutionResult
```

## Key Modules

### PipelineRunner
**File:** `orchestrator/src/orchestrator/pipeline/runner.py`

| Method | Purpose | Returns |
|--------|---------|---------|
| `execute(symbol, timeframe, model_override)` | Main pipeline entry point, orchestrates all stages | `PipelineResult` |
| `_run_agents(data)` | Calls sentiment, market, proposer in parallel | `(SentimentReport, MarketInterpretation, TradeProposal)` |
| `_check_risk_and_store(proposal)` | Risk check + database storage | `RiskResult` |
| `_await_approval(proposal)` | Sends Telegram, waits for response | `ApprovalRecord` |

Key fields in `PipelineResult`:
- `run_id` — Unique identifier for pipeline run, bound to all logs
- `status` — "completed", "rejected", "risk_rejected", "risk_paused", "pending_approval", "failed"
- `proposal` — The generated TradeProposal (null if rejected)
- `approval_id` — Reference to Telegram approval (null if no approval needed)
- `sentiment_degraded`, `market_degraded`, `proposer_degraded` — Agent fallback flags

### DataFetcher
**File:** `orchestrator/src/orchestrator/exchange/data_fetcher.py`

| Method | Purpose |
|--------|---------|
| `fetch_snapshot(symbol, timeframe)` | Fetches OHLCV (10 candles), current price, funding rate, 24h volume |
| `fetch_ohlcv(symbol, timeframe)` | Low-level CCXT OHLCV fetch with retry logic |

Returns `MarketSnapshot`:
```python
class MarketSnapshot:
    symbol: str
    current_price: float
    ohlcv: list[tuple[float, float, float, float, float]]  # [open, high, low, close, volume]
    volume_24h: float
    funding_rate: float  # Perpetual 8h rate
    timeframe: str
```

### Agent Base Class
**File:** `orchestrator/src/orchestrator/agents/base.py`

`BaseAgent[T]` is a generic async agent that:
1. Builds messages from snapshot data
2. Calls LLM with skill-based prompt
3. Validates output against Pydantic model (T)
4. Retries on validation failure with error feedback
5. Degrades to default output if max retries exhausted

Key methods:
- `analyze(**kwargs)` — Async entry point
- `_build_messages(**kwargs)` — Formats prompt (must override)
- `_get_default_output()` — Fallback when agent fails (must override)

### Sentiment Agent
**File:** `orchestrator/src/orchestrator/agents/sentiment.py`

- **Skill:** `.claude/skills/sentiment/SKILL.md`
- **Input:** MarketSnapshot
- **Output:** `SentimentReport` (score 0-100, key_events, confidence 0.0-1.0)
- **Prompt:** Includes OHLCV summary (10 candles), funding rate, volume

### Market Agent
**File:** `orchestrator/src/orchestrator/agents/market.py`

- **Skill:** `.claude/skills/market/SKILL.md`
- **Input:** MarketSnapshot
- **Output:** `MarketInterpretation` (trend, volatility_regime, key_levels, risk_flags)
- **Prompt:** Price action analysis, support/resistance detection

### Proposer Agent
**File:** `orchestrator/src/orchestrator/agents/proposer.py`

- **Skill:** `.claude/skills/proposer/SKILL.md`
- **Input:** MarketSnapshot, SentimentReport, MarketInterpretation
- **Output:** `TradeProposal` (side, entry, SL, TP, position_size_risk_pct, leverage)
- **Prompt:** Combines all three inputs, applies decision framework

### Aggregator
**File:** `orchestrator/src/orchestrator/pipeline/aggregator.py`

```python
def aggregate_proposal(
    proposal: TradeProposal,
    sentiment: SentimentReport,
    market: MarketInterpretation,
) -> TradeProposal:
    """Validates proposal against sentiment/market, returns adjusted copy or flat."""
```

Checks:
- If proposal.side doesn't match sentiment direction → flatten
- If multiple risk_flags → reduce position size
- Confidence < 0.3 → flatten

### Pipeline Scheduler
**File:** `orchestrator/src/orchestrator/pipeline/scheduler.py`

APScheduler wrapper that:
- Triggers pipeline every N minutes for configured symbols
- Handles task cancellation on shutdown
- Logs job execution

Entry point: `scheduler.start()` → runs until `scheduler.shutdown()`

## Data Flow

### Input
- Symbol (e.g., "BTC/USDT:USDT")
- Timeframe (default "1h")
- Optional model override (e.g., "anthropic/claude-opus-4-6")

### Processing
1. **Fetch data** — CCXT snapshot with 10 recent candles
2. **Analyze sentiment** — Score market sentiment 0-100
3. **Analyze market** — Identify trend, volatility, key levels
4. **Propose trade** — Decision framework to enter long/short/flat
5. **Aggregate** — Cross-validate proposal against sentiment/market
6. **Check risk** — Compare against position limits, daily loss, consecutive losses
7. **Await approval** — Telegram inline button approval (semi-auto)
8. **Execute** — Paper or live order execution

### Output
`PipelineResult` containing:
- Final proposal (may be null if rejected/paused)
- All agent outputs (for logging/analysis)
- Approval reference
- Risk result and decision
- Degradation flags if agents failed

## Testing

Unit tests cover:
- Individual agent outputs
- Aggregator logic
- Risk checker rules
- Paper trading flow

**Test files:**
- `tests/unit/test_agent_sentiment.py`
- `tests/unit/test_agent_proposer.py`
- `tests/unit/test_paper_trading_flow.py`
- `tests/unit/test_risk_checker.py`

## Logging

All pipeline events logged to structlog with `run_id` context:
```python
structlog.contextvars.bind_contextvars(run_id=run_id, symbol=symbol)
logger.info("pipeline_start")
logger.info("agent_success", agent="SentimentAgent")
logger.warning("risk_rejected", rule="max_single_risk")
```

View logs:
```bash
uv run python -m orchestrator 2>&1 | grep "run_id=..."
```

## Configuration

Environment variables affecting pipeline:
- `PIPELINE_SYMBOLS` — Comma-separated list (default: "BTC/USDT:USDT,ETH/USDT:USDT")
- `PIPELINE_INTERVAL_MINUTES` — Run frequency (default: 15)
- `LLM_MODEL` — Standard model for agents (default: "anthropic/claude-sonnet-4-6")
- `LLM_MODEL_PREMIUM` — Used if model_override="opus"
- `LLM_TEMPERATURE` — Agent temperature (default: 0.2)
- `LLM_MAX_TOKENS` — Max response length (default: 2000)
- `LLM_MAX_RETRIES` — Agent validation retries (default: 1)

See `orchestrator/src/orchestrator/config.py` for full list.

## Related

- [agents-skills.md](agents-skills.md) — Detailed agent & skill architecture
- [risk.md](risk.md) — Risk checking logic
- [approval-telegram.md](approval-telegram.md) — Approval UI & semi-auto flow
- [execution.md](execution.md) — Order execution details
