<!-- Generated: 2026-03-02 | Files scanned: 8 | Token estimate: ~700 -->
# Pipeline Codemap

## Overview

The pipeline fetches market + external data, runs 5 analysis agents in parallel, feeds results to Proposer, then validates through aggregation and risk checks.

## Architecture

```
PipelineRunner.execute(symbol, timeframe)
    │
    ├─ Step 1: Fetch all data in parallel (asyncio.gather)
    │   ├─ DataFetcher.fetch_snapshot → OHLCV, price, funding
    │   ├─ DataFetcher.fetch_positioning_data → OI, L/S ratios, order book
    │   ├─ DataFetcher.fetch_macro_indicators → 200W MA, bull support band
    │   ├─ ExternalDataFetcher.fetch_dxy_data → DXY index
    │   ├─ ExternalDataFetcher.fetch_sp500_data → S&P 500
    │   ├─ ExternalDataFetcher.fetch_btc_dominance → BTC.D
    │   ├─ ExternalDataFetcher.fetch_economic_calendar
    │   └─ ExternalDataFetcher.fetch_exchange_announcements
    │
    ├─ Step 2: Run 5 analysis agents in parallel (asyncio.gather)
    │   ├─ TechnicalAgent(short) → TechnicalAnalysis
    │   ├─ TechnicalAgent(long) → TechnicalAnalysis
    │   ├─ PositioningAgent → PositioningAnalysis
    │   ├─ CatalystAgent → CatalystReport
    │   └─ CorrelationAgent → CorrelationAnalysis
    │
    ├─ Step 3: ProposerAgent → TradeProposal (depends on all 5)
    │
    ├─ Step 4: aggregate_proposal → validation
    │
    ├─ Step 5: RiskChecker.check → approve / reject / pause
    │
    ├─ Step 6: ApprovalManager or auto-execute
    │
    └─ Step 7: Save & return PipelineResult
```

## Key Files

### PipelineRunner — `pipeline/runner.py`

| Method | Purpose |
|--------|---------|
| `execute(symbol, timeframe, model_override)` | Main entry, orchestrates all steps |
| `_build_result(...)` | Constructs PipelineResult with all agent degradation flags |
| `_save_llm_calls(run_id, agent_type, result)` | Persists LLM calls to DB |

`PipelineResult` fields:
- `status` — completed, rejected, risk_rejected, risk_paused, pending_approval, failed
- `technical_short`, `technical_long`, `positioning`, `catalyst`, `correlation` — agent outputs
- `*_degraded` — per-agent fallback flags (5 analysis + proposer)
- `risk_result`, `proposal`, `approval_id`

Error handling: catches all exceptions, logs `pipeline_failed` with `error_type` and `exc_info`.

### Aggregator — `pipeline/aggregator.py`

```python
aggregate_proposal(proposal, current_price) → AggregationResult
```
Validates proposal fields, returns `valid=True/False` with `rejection_reason`.

### Scheduler — `pipeline/scheduler.py`

APScheduler wrapper:
- `run_once(symbols, model_override, source)` — runs all symbols in parallel via `asyncio.gather`
- Interval trigger: every N minutes (default model)
- Cron trigger: daily at 00:00 UTC (premium model)
- Approval expiry check: every 1 minute

## Logging Events

Pipeline lifecycle (all bound to `run_id` + `symbol`):
```
pipeline_start → snapshot_fetched → agent_start(×5) → agent_success(×5)
  → agent_start(proposer) → agent_success(proposer)
  → pipeline_completed | pipeline_rejected | pipeline_risk_blocked | pipeline_pending_approval
  → pipeline_done
```

On failure: `pipeline_failed` with `error`, `error_type`, full traceback.

## Testing

- `tests/unit/test_pipeline_runner_v2.py` — Runner with mocked agents
- `tests/unit/test_runner.py` — Pipeline execution paths
- `tests/unit/test_scheduler.py` — Scheduler interval/cron logic
- `tests/unit/test_aggregator.py` — Proposal validation
- `tests/integration/test_pipeline_integration.py` — Full pipeline with mocked LLM
