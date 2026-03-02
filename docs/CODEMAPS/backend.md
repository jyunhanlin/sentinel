<!-- Generated: 2026-03-02 | Files scanned: 50 | Token estimate: ~950 -->

# Backend — Pipeline & Agent Architecture

## Entry Point

`orchestrator/__main__.py` (392L)
- `parse_args()` → subcommands: `eval`, `perf`, default=bot
- `create_app_components(**kwargs)` → manual DI, returns component dict
- `_run_bot(components, settings)` → starts scheduler + Telegram bot

## Pipeline

### PipelineRunner (`pipeline/runner.py`, 385L)

```
execute(symbol, timeframe, model_override) -> PipelineResult
  1. asyncio.gather: snapshot, positioning, macro, DXY, SP500, BTC.D, calendar, announcements
  2. asyncio.gather: technical_short, technical_long, positioning, catalyst, correlation
  3. serial:         proposer(all_analyses) → TradeProposal
  4. aggregate_proposal(proposal, price) → validates SL placement
  5. risk_checker.check(proposal, exposure, losses, daily_loss)
  6. approval_manager.create() or paper_engine.open_position()
  7. persist: pipeline_run + llm_calls + proposal
```

**PipelineResult.status**: `completed | rejected | risk_rejected | risk_paused | pending_approval | failed`

### PipelineScheduler (`pipeline/scheduler.py`, 143L)

| Job | Interval | Model |
|-----|----------|-------|
| `run_once()` | every N min (default 720) | Sonnet |
| `_run_daily_premium()` | daily 00:00 UTC | Opus |
| `_expire_stale_approvals()` | every 1 min | — |
| `price_monitor.check()` | every N sec | — |

### Aggregator (`pipeline/aggregator.py`, 48L)

`aggregate_proposal(proposal, current_price) -> AggregationResult`
- Validates SL exists for directional trades
- Validates SL side (long: SL < price, short: SL > price)

## Agent System

### BaseAgent[T] (`agents/base.py`, 123L)

```
analyze(**kwargs) -> AgentResult[T]
  → _build_messages(**kwargs) → [{"role":"user", "content": _build_prompt()}]
  → llm_client.call(messages) → LLMCallResult
  → validate_llm_output(raw, output_model) → T
  → retry up to max_retries with error feedback
  → on failure: _get_default_output() + degraded=True
```

### Agent Implementations

| Agent | File | Output Model | Key Input |
|-------|------|-------------|-----------|
| TechnicalAgent | technical.py (72L) | TechnicalAnalysis | MarketSnapshot, macro_data |
| PositioningAgent | positioning.py (57L) | PositioningAnalysis | funding, OI, L/S ratios |
| CatalystAgent | catalyst.py (55L) | CatalystReport | calendar, announcements |
| CorrelationAgent | correlation.py (51L) | CorrelationAnalysis | DXY, SP500, BTC.D |
| ProposerAgent | proposer.py (104L) | TradeProposal | all 5 analyses + snapshot |

**Two TechnicalAgent instances:** short_term (50 candles), long_term (30 candles + macro)

## LLM Layer

### LLMBackend (`llm/backend.py`, 205L)

| Backend | How |
|---------|-----|
| `LiteLLMBackend` | `litellm.acompletion()` with API key |
| `ClaudeCLIBackend` | spawns `claude -p --output-format json`, strips `ANTHROPIC_API_KEY` |

### LLMClient (`llm/client.py`, 61L)

`call(messages, model, temperature, max_tokens) -> LLMCallResult`

### SchemaValidator (`llm/schema_validator.py`, 61L)

`validate_llm_output(raw, model_class)` → extracts JSON (handles ``` blocks) → Pydantic validates

## Risk Management

### RiskChecker (`risk/checker.py`, 100L)

| Rule | Action |
|------|--------|
| Single risk % > limit | reject |
| Total exposure % > limit | reject |
| Consecutive losses >= limit | pause |
| Daily loss % > limit | pause |

### Position Sizers (`risk/position_sizer.py`, 39L)

- `RiskPercentSizer`: qty = (equity × risk%) / |entry − SL|
- `MarginSizer`: qty = margin × leverage / entry

## Execution

### OrderExecutor (`execution/executor.py`, 224L)

| Executor | Behavior |
|----------|----------|
| `PaperExecutor` | delegates to PaperEngine |
| `LiveExecutor` | exchange market order + SL/TP stop orders, price deviation check |

## Approval Flow

### ApprovalManager (`approval/manager.py`, 123L)

```
create(proposal) → PendingApproval (in-memory + DB)
approve(id) → checks expiry → marks approved
reject(id) → marks rejected
expire_stale() → called every 1 min by scheduler
```
