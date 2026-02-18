# M1 Design: 3-Model Pipeline

**Date:** 2026-02-18
**Status:** Approved
**Depends on:** M0 (project skeleton)

---

## Goal

Build the 3-LLM agent pipeline that analyzes market data, generates structured trade proposals, and pushes results through Telegram. This is the core intelligence layer of Sentinel.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sentiment data source | LLM built-in knowledge + MarketSnapshot | MVP fastest path; interface designed for future multi-source expansion |
| LLM failure handling | Retry 1x + degrade to neutral defaults | Balances reliability with pipeline continuity |
| Scheduler | APScheduler interval + TG `/run` manual trigger | Automated + easy testing/demo |
| TG push behavior | Push every pipeline run (including FLAT) | Full visibility into system operation |
| Model selection | Dual-model: Sonnet (scheduled) + Opus (manual/daily) | Cost vs. depth tradeoff; Sonnet for frequent automated runs, Opus for deliberate analysis |

## Architecture

### New Modules (on top of M0)

```
llm/
  client.py              LiteLLM wrapper with cost tracking
  schema_validator.py    JSON string → Pydantic model + retry error messages

agents/
  base.py                BaseAgent ABC with retry + degrade
  sentiment.py           LLM-1: market sentiment analysis
  market.py              LLM-2: technical/order flow interpretation
  proposer.py            LLM-3: trade proposal generation

pipeline/
  runner.py              Single pipeline run orchestration
  aggregator.py          Merge + validate 3 agent outputs
  scheduler.py           APScheduler periodic trigger

telegram/
  bot.py                 + /run command
  formatters.py          + format_proposal()
```

### Data Flow

```
Scheduler / TG /run
    │
    ▼
Runner.execute(symbol)
    ├── DataFetcher.fetch_snapshot()          ← M0
    │
    ├── [asyncio.gather — parallel]
    │   ├── SentimentAgent.analyze(snapshot)  → SentimentReport
    │   └── MarketAgent.analyze(snapshot)     → MarketInterpretation
    │
    ├── ProposerAgent.propose(sentiment, market, snapshot) → TradeProposal
    │
    ├── Aggregator.validate(proposal)         → validated TradeProposal
    │
    ├── Storage.save(run, llm_calls, proposal) ← M0
    │
    └── TG Push (formatted proposal message)
```

## Component Details

### LLM Client (`llm/client.py`)

- Wraps LiteLLM `acompletion()` for unified multi-provider interface
- Method: `async call(messages, *, model=None, ...) → LLMCallResult`
- Per-call `model` override: allows switching between Sonnet/Opus without creating separate client instances
- Records every call to `LLMCallRecord`: model, latency_ms, tokens, cost
- Configurable `temperature` and `max_tokens`

### Schema Validator (`llm/schema_validator.py`)

- Input: raw LLM string output
- Process: strip non-JSON text → parse JSON → validate against Pydantic model
- On success: return validated model instance
- On failure: return structured error message (for retry prompt)
- Degrade support: return neutral default + warning flag

### BaseAgent (`agents/base.py`)

Abstract base class providing:
- `async analyze(**kwargs) → T` public method with retry + degrade
- `_build_prompt(**kwargs) → list[dict]` — subclass implements
- `_get_default_output() → T` — subclass implements (neutral fallback)
- `_parse_output(raw: str) → T` — uses schema_validator
- Retry logic: 1 retry with error feedback in prompt, then degrade

### Agents

| Agent | Input | Output | Strategy |
|-------|-------|--------|----------|
| SentimentAgent | MarketSnapshot | SentimentReport | Infer sentiment from price action, funding rate, volume patterns using LLM knowledge |
| MarketAgent | MarketSnapshot | MarketInterpretation | Analyze OHLCV for trend, volatility regime, key support/resistance levels, risk flags |
| ProposerAgent | SentimentReport + MarketInterpretation + MarketSnapshot | TradeProposal | Synthesize both analyses into directional trade with entry/SL/TP/sizing |

### Pipeline Runner (`pipeline/runner.py`)

- Generates `run_id` (UUID) per execution
- Binds `run_id` to structlog context
- Parallel execution of LLM-1 + LLM-2 via `asyncio.gather`
- Sequential LLM-3 (depends on LLM-1 + LLM-2)
- Aggregator validates final proposal
- Persists all records via repository
- Returns `PipelineResult` with run status + proposal

### Aggregator (`pipeline/aggregator.py`)

- Validates TradeProposal completeness
- Sanity checks: SL on correct side of entry, confidence in range
- Attaches metadata: run_id, timestamp
- Returns validated proposal or rejection reason

### Scheduler (`pipeline/scheduler.py`)

- APScheduler `AsyncIOScheduler` with `IntervalTrigger`
- Iterates `pipeline_symbols` list, runs each through pipeline
- Graceful shutdown on SIGTERM/SIGINT
- Configurable interval via `Settings.pipeline_interval_minutes`
- **Dual schedule:**
  - Regular interval (e.g., every 15min) → Sonnet (`llm_model`)
  - Daily deep analysis (e.g., 00:00 UTC) → Opus (`llm_model_premium`)

### Model Selection Strategy

| Trigger | Default Model | Override? |
|---------|---------------|-----------|
| Scheduled interval (15min) | Sonnet | No |
| Daily deep analysis | Opus | No |
| TG `/run` (manual) | Opus | Yes — `/run BTC sonnet` |

- `LLMClient.call()` accepts per-call `model` override
- `BaseAgent.analyze()` passes `model_override` down to client
- `PipelineRunner.execute()` accepts `model_override` parameter
- TG `/run` defaults to Opus (intentional trigger = want deeper analysis)
- User can downgrade via `/run [symbol] sonnet` or `/run [symbol] opus`

### Telegram Enhancements

- `/run` command: manually triggers pipeline for all configured symbols (default: Opus)
- `/run BTC` variant: trigger for specific symbol (default: Opus)
- `/run BTC sonnet` variant: trigger with explicit model choice
- `/status`: shows latest pipeline run results per symbol
- `/coin BTC`: shows most recent proposal for that symbol
- `format_proposal()`: structured message with direction, entry, SL/TP, confidence, rationale, **model used**

## Future: Strategy Layer (Post-M1)

**Decision:** A+B 混合模式

```
Strategy.compute() → signals (structured data)
    │
    ▼
[Sentiment + Market + Signals] → ProposerAgent → TradeProposal
                                                      │
                                                      ▼
                                              StrategyGate.check()
                                              ├── PASS → TG Push
                                              └── CONFLICT → TG Push (標註衝突)
```

- **Pre-Proposer:** Strategy signals 作為 ProposerAgent 的額外 context
- **Post-Proposer:** StrategyGate 做 code-level 硬檢查，衝突時標註但仍推送
- **Pluggable:** strategies/ 目錄下可放多個策略，config 控制啟用哪些
- 第一個策略範例：Turtle Head Strategy（熊市假突破做空）

## Out of Scope

- Risk check engine (M2)
- Paper trading execution (M2)
- Real news/social API integration (future)
- Approval flow for proposals (M4)
- Multi-exchange support
