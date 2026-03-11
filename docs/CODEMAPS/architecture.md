<!-- Generated: 2026-03-11 | Files scanned: 49 | Token estimate: ~900 -->

# Architecture — Sentinel Orchestrator

## System Overview

Async Python 3.12+ crypto futures trading orchestrator.
5+1+1 LLM agent pipeline → critic/refinement → semi-auto Telegram approval → paper/live execution.

```
               ┌─────────────────────────────────────────┐
               │              __main__.py                 │
               │  (composition root, manual DI wiring)    │
               └──────────┬──────────────────────────────┘
                          │
          ┌───────────────┼───────────────────┐
          ▼               ▼                   ▼
   PipelineScheduler   SentinelBot      PriceMonitor
   (APScheduler)       (Telegram)       (SL/TP checker)
          │               ▲                   │
          ▼               │ callbacks         ▼
   PipelineRunner ────────┘            PaperEngine
          │
    ┌─────┼──────────────────────────────┐
    ▼     ▼                              ▼
  Agents(5)  →  RefinementLoop        ApprovalManager
  (parallel)    (Proposer ↔ Critic)        │
                       │                   ▼
                       └──────────→  OrderExecutor
                                  (Paper | Live)
```

## Pipeline Flow (per symbol)

1. **Data fetch** — `DataFetcher` + `ExternalDataFetcher` (parallel `asyncio.gather`)
2. **5-agent analysis** — technical×2, positioning, catalyst, correlation (parallel)
3. **Proposer** — synthesizes all analyses into `TradeProposal` (serial)
4. **Critic** — validates proposal across 4 dimensions (serial)
5. **Refinement** — if critic rejects, proposer revises with feedback (up to N rounds)
6. **Aggregation** — validates SL placement
7. **Execution planning** — `ExecutionPlanner` computes quantities, leverage, fees
8. **Approval** — Telegram inline keyboard (approve → leverage → margin → execute)
9. **Persistence** — SQLite via SQLModel repositories

## Callback Wiring

```
scheduler._on_result     → bot.push_to_admins_with_approval
price_monitor._on_close  → bot.push_close_report
price_monitor._on_tick   → bot.update_price_board
```

## Key Patterns

- **Immutable models**: all Pydantic `frozen=True`
- **Async-first**: `asyncio.gather` for parallel I/O
- **Repository pattern**: 6 repo classes wrap all DB access
- **Strategy pattern**: LLM backends (CLI/API), position sizers (risk%/margin)
- **Degraded mode**: agents return safe defaults on LLM failure
- **Skill-based prompting**: agents load `.claude/skills/{name}/SKILL.md`

## Module Map (8,067 lines across 49 files)

| Package | Purpose | Key files |
|---------|---------|-----------|
| `agents/` | LLM analysis agents | base.py, technical.py, positioning.py, catalyst.py, correlation.py, proposer.py, critic.py |
| `pipeline/` | Orchestration & scheduling | runner.py (390L), scheduler.py (146L), refinement.py (129L), aggregator.py (48L) |
| `exchange/` | Market data & paper trading | client.py, data_fetcher.py, paper_engine.py (688L), price_monitor.py |
| `llm/` | LLM abstraction | backend.py (205L), client.py, schema_validator.py |
| `execution/` | Order execution + position sizing | executor.py (224L), planner.py (141L), position_sizer.py, plan.py, equity.py |
| `storage/` | SQLite persistence | repository.py (443L), models.py, database.py, migrations.py (187L) |
| `telegram/` | Bot UI & notifications | bot.py (2211L), formatters.py (838L), translations.py |
| `approval/` | Trade approval lifecycle | manager.py (123L) |
| `stats/` | Performance metrics + pipeline evaluation | calculator.py (101L), evaluator.py (224L) |
