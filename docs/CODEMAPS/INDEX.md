# Sentinel Codemaps Index

**Last Updated:** 2026-02-28

This document maps all features and domains to their code locations in the Sentinel trading orchestrator.

## Quick Navigation

| Domain | Purpose | Codemap |
|--------|---------|---------|
| **Pipeline** | Market analysis + trade proposal execution flow | [pipeline.md](pipeline.md) |
| **Agents & Skills** | LLM agents + skill-based prompts for analysis | [agents-skills.md](agents-skills.md) |
| **Risk Management** | Position sizing, risk checks, daily loss limits | [risk.md](risk.md) |
| **Execution** | Trade order execution (paper + live mode) | [execution.md](execution.md) |
| **Storage** | SQLite database models, migrations, repositories | [storage.md](storage.md) |
| **Exchange** | Data fetching, paper trading engine, price monitor | [exchange.md](exchange.md) |
| **Approval & Telegram** | Semi-auto trading with inline approval buttons | [approval-telegram.md](approval-telegram.md) |
| **Evaluation** | LLM output validation, golden dataset testing | [evaluation.md](evaluation.md) |
| **Configuration** | Environment variables, settings, constants | [configuration.md](configuration.md) |

## Architecture Overview

```
Scheduler (APScheduler)
    │
    ├─ PipelineRunner
    │   ├── DataFetcher (CCXT, OHLCV)
    │   ├── SentimentAgent (skill: sentiment)
    │   ├── MarketAgent (skill: market)
    │   ├── ProposerAgent (skill: proposer)
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
- **`orchestrator/CLAUDE.md`** — Project conventions (immutability, async-first, structured logging)

### Core Models
- **`orchestrator/src/orchestrator/models.py`** — Domain models (Side, Trend, TradeProposal, SentimentReport, etc.)
- **`orchestrator/src/orchestrator/config.py`** — Pydantic Settings (LLM, exchange, risk params)

### Skills (Prompt Templates)
- **`.claude/skills/sentiment/SKILL.md`** — Sentiment analysis methodology
- **`.claude/skills/market/SKILL.md`** — Technical analysis (trend, volatility, levels)
- **`.claude/skills/proposer/SKILL.md`** — Trade proposal generation logic

### Database
- **`orchestrator/src/orchestrator/storage/models.py`** — SQLModel tables (PipelineRunRecord, TradeProposalRecord, etc.)
- **`orchestrator/src/orchestrator/storage/database.py`** — SQLite setup, migrations

### Telegram Bot
- **`orchestrator/src/orchestrator/telegram/bot.py`** — Command handlers, callback handlers
- **`orchestrator/src/orchestrator/telegram/formatters.py`** — Message formatting (status, proposals, reports)

## Development Commands

```bash
# Run entire pipeline once
cd orchestrator
uv run python -m orchestrator

# Run tests
uv run pytest -v --cov=orchestrator

# Lint + format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# LLM evaluation
uv run python -m orchestrator eval

# Performance report
uv run python -m orchestrator perf
```

## Data Flow

### Pipeline Execution (every N minutes)
1. **DataFetcher** → fetches OHLCV, current price, funding rate
2. **SentimentAgent** → calls "sentiment" skill → SentimentReport
3. **MarketAgent** → calls "market" skill → MarketInterpretation
4. **ProposerAgent** → calls "proposer" skill → TradeProposal
5. **RiskChecker** → validates proposal against limits
6. **ApprovalManager** → sends Telegram approval request
7. **OrderExecutor** → executes on user approval

### Approval Flow
- Proposal published with Approve/Reject buttons
- User has 15 minutes (configurable) to respond
- Price checked for staleness (>1% deviation)
- If expired or rejected → pipeline continues, no trade
- If approved → OrderExecutor fills entry, places SL/TP

### Closing Trades
- **PriceMonitor** → watches open positions every 5 minutes
- **PaperEngine** → simulates SL/TP fills
- **StatsCalculator** → computes PnL, win rate, Sharpe ratio

## Key Patterns

### Immutability
All Pydantic models use `frozen=True`. New copies created via:
```python
updated = model.model_copy(update={"field": new_value})
```

### Repository Pattern
Data access through repositories, not direct DB queries:
- `PipelineRepository`, `TradeProposalRepository`, `PaperTradeRepository`
- Enables easy testing with mocks

### Async-First
LLM calls and exchange data fetches use `asyncio`:
```python
sentiment_result = await sentiment_agent.analyze(snapshot=snapshot)
```

### Skill-Based Prompts
Instead of embedded prompts, agents reference external skill files:
```python
message = f"Use the {self._skill_name} skill.\n\n=== Market Data ===\n{data}"
```
Claude reads `.claude/skills/{skill_name}/SKILL.md` and applies methodology.

### Structured Logging
All logs include `run_id` for pipeline tracing:
```python
structlog.contextvars.bind_contextvars(run_id=run_id, symbol=symbol)
logger.info("pipeline_start", run_id=run_id)
```

## Recent Features (Feb 2026)

- **Skills-First Architecture** — Prompts moved to `.claude/skills/` for better separation of concerns
- **Paper Trading** — Full simulation with margin, leverage, fees
- **Price Monitor** — Async SL/TP tracking with partial fills
- **Performance Stats** — Sharpe ratio, max drawdown, win rate calculations
- **Agent Leverage** — Per-proposal leverage override (1-50x)
- **CLI Backend Selection** — Switch between LiteLLM API and Claude CLI

See `docs/plans/` for detailed design docs.
