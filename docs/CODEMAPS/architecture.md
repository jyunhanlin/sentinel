<!-- Generated: 2026-03-02 | Files scanned: 50 | Token estimate: ~900 -->

# Architecture — Sentinel Orchestrator

## System Overview

Async Python 3.12+ crypto futures trading orchestrator.
5+1 LLM agent pipeline → risk gate → semi-auto Telegram approval → paper/live execution.

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
    ┌─────┼─────────────────────┐
    ▼     ▼                     ▼
  Agents(5)  →  ProposerAgent  →  RiskChecker
  (parallel)    (serial)           │
                                   ▼
                            ApprovalManager
                                   │
                                   ▼
                            OrderExecutor
                            (Paper | Live)
```

## Pipeline Flow (per symbol)

1. **Data fetch** — `DataFetcher` + `ExternalDataFetcher` (parallel `asyncio.gather`)
2. **5-agent analysis** — technical×2, positioning, catalyst, correlation (parallel)
3. **Proposer** — synthesizes all analyses into `TradeProposal` (serial)
4. **Aggregation** — validates SL placement
5. **Risk check** — 4 rules (single risk, total exposure, consecutive losses, daily loss)
6. **Approval** — Telegram inline keyboard (approve → leverage → margin → execute)
7. **Persistence** — SQLite via SQLModel repositories

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

## Module Map (7,534 lines across 50 files)

| Package | Purpose | Key files |
|---------|---------|-----------|
| `agents/` | LLM analysis agents | base.py, technical.py, positioning.py, catalyst.py, correlation.py, proposer.py |
| `pipeline/` | Orchestration & scheduling | runner.py (385L), scheduler.py (143L), aggregator.py (48L) |
| `exchange/` | Market data & paper trading | client.py, data_fetcher.py, paper_engine.py (677L), price_monitor.py |
| `llm/` | LLM abstraction | backend.py (205L), client.py, schema_validator.py |
| `risk/` | Risk management | checker.py (100L), position_sizer.py (39L) |
| `storage/` | SQLite persistence | repository.py (441L), models.py, database.py, migrations.py |
| `telegram/` | Bot UI & notifications | bot.py (1483L), formatters.py (718L), translations.py |
| `approval/` | Trade approval lifecycle | manager.py (123L) |
| `execution/` | Order execution | executor.py (224L) |
| `stats/` | Performance metrics | calculator.py (101L) |
| `eval/` | LLM evaluation | runner.py, dataset.py, scorers.py, consistency.py |
