<!-- Generated: 2026-03-02 | Files scanned: 51 | Token estimate: ~900 -->
# Sentinel Codemaps Index

## Quick Navigation

| Domain | Purpose | Codemap |
|--------|---------|---------|
| **Pipeline** | 5-agent analysis + trade proposal execution flow | [pipeline.md](pipeline.md) |
| **Agents & Skills** | LLM agents + skill-based prompts for analysis | [agents-skills.md](agents-skills.md) |
| **Risk Management** | Position sizing, risk checks, daily loss limits | [risk.md](risk.md) |
| **Execution** | Trade order execution (paper + live mode) | [execution.md](execution.md) |
| **Storage** | SQLite database models, migrations, repositories | [storage.md](storage.md) |
| **Exchange** | Data fetching, external data, paper engine, price monitor | [exchange.md](exchange.md) |
| **Approval & Telegram** | Semi-auto trading with inline approval buttons | [approval-telegram.md](approval-telegram.md) |
| **Evaluation** | LLM output validation, golden dataset testing | [evaluation.md](evaluation.md) |
| **Configuration** | Environment variables, settings, constants | [configuration.md](configuration.md) |

## Architecture Overview

```
Scheduler (APScheduler)
    │
    ├─ PipelineRunner
    │   ├── DataFetcher (CCXT — OHLCV, funding, OI, L/S ratio, order book)
    │   ├── ExternalDataFetcher (Yahoo Finance — DXY, S&P 500; CoinGecko — BTC.D)
    │   ├── TechnicalAgent ×2 (short_term 50 candles, long_term 30 candles)
    │   ├── PositioningAgent (funding, OI, L/S ratios, order book)
    │   ├── CatalystAgent (economic calendar, exchange announcements)
    │   ├── CorrelationAgent (DXY, S&P 500, BTC dominance)
    │   ├── ProposerAgent (synthesizes all 5 → TradeProposal)
    │   ├── Aggregator
    │   └── RiskChecker
    │
    ├─ ApprovalManager
    │   ├── Telegram Bot (inline buttons)
    │   └── Price deviation check
    │
    ├─ OrderExecutor
    │   ├── PaperExecutor (PaperEngine)
    │   └── LiveExecutor (ExchangeClient)
    │
    ├─ PriceMonitor (SL/TP tracking)
    │
    └─ Stats Calculator (PnL, win rate, Sharpe)
```

## Key Files & Entry Points

### Application Entry
- **`orchestrator/src/orchestrator/__main__.py`** — CLI entry, component factory, scheduler startup
- **`CLAUDE.md`** — Project conventions (immutability, async-first, structured logging)

### Core Models
- **`orchestrator/src/orchestrator/models.py`** — Domain models (Side, Trend, TechnicalAnalysis, PositioningAnalysis, CatalystReport, CorrelationAnalysis, TradeProposal)
- **`orchestrator/src/orchestrator/config.py`** — Pydantic Settings (LLM, exchange, risk params)

### Skills (Prompt Templates)
- **`.claude/skills/technical/SKILL.md`** — Technical analysis (trend, volatility, momentum, key levels)
- **`.claude/skills/positioning/SKILL.md`** — Futures positioning & order flow
- **`.claude/skills/catalyst/SKILL.md`** — Event & news catalyst analysis
- **`.claude/skills/correlation/SKILL.md`** — Cross-market correlation (DXY, S&P, BTC.D)
- **`.claude/skills/proposer/SKILL.md`** — Trade proposal generation logic

### Database
- **`orchestrator/src/orchestrator/storage/models.py`** — SQLModel tables
- **`orchestrator/src/orchestrator/storage/database.py`** — SQLite setup, migrations

### Telegram Bot
- **`orchestrator/src/orchestrator/telegram/bot.py`** — Command handlers, callback handlers
- **`orchestrator/src/orchestrator/telegram/formatters.py`** — Message formatting

## Data Flow

### Pipeline Execution (every N minutes)
1. **DataFetcher** → OHLCV, price, funding, OI, L/S ratio, order book
2. **ExternalDataFetcher** → DXY, S&P 500, BTC dominance, economic calendar
3. **5 Analysis Agents (parallel):**
   - TechnicalAgent (short) → TechnicalAnalysis
   - TechnicalAgent (long) → TechnicalAnalysis
   - PositioningAgent → PositioningAnalysis
   - CatalystAgent → CatalystReport
   - CorrelationAgent → CorrelationAnalysis
4. **ProposerAgent** → TradeProposal (consumes all 5 outputs)
5. **Aggregator** → validates proposal
6. **RiskChecker** → validates against limits
7. **ApprovalManager** → sends Telegram approval request
8. **OrderExecutor** → executes on user approval

## Key Patterns

### Immutability
All Pydantic models use `frozen=True`. New copies via `model.model_copy(update={...})`.

### Repository Pattern
Data access through repositories, not direct DB queries.

### Async-First
LLM calls and exchange data fetches use `asyncio`. 5 analysis agents run in parallel via `asyncio.gather`.

### Skill-Based Prompts
Agents reference external skill files: `f"Use the {self._skill_name} skill.\n\n=== Data ===\n{data}"`

### Structured Logging
All logs include `run_id`/`symbol` via structlog contextvars. Agent logs include `agent` name and `label` (for TechnicalAgent short/long).
