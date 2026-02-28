# Sentinel Codemaps

This directory contains comprehensive architecture documentation mapping every feature and domain to its code locations.

## Start Here

**New to the codebase?** Read in this order:

1. **[INDEX.md](INDEX.md)** — Overview of all domains and architecture
2. **[pipeline.md](pipeline.md)** — Main execution flow (sentiment → market → proposer → execution)
3. **[agents-skills.md](agents-skills.md)** — LLM agents and skill-based prompts
4. **[configuration.md](configuration.md)** — Environment setup

## By Domain

| Codemap | Covers |
|---------|--------|
| [pipeline.md](pipeline.md) | Pipeline execution, DataFetcher, PipelineRunner, scheduler |
| [agents-skills.md](agents-skills.md) | Agent base class, sentiment/market/proposer agents, skill files, LLM validation |
| [risk.md](risk.md) | RiskChecker, position sizing, paper engine margin/liquidation, stats |
| [execution.md](execution.md) | OrderExecutor, PaperExecutor, LiveExecutor, SL/TP placement |
| [storage.md](storage.md) | SQLModel tables, repositories, database setup, migrations |
| [exchange.md](exchange.md) | ExchangeClient, DataFetcher, PaperEngine, PriceMonitor |
| [approval-telegram.md](approval-telegram.md) | ApprovalManager, Telegram bot, commands, inline buttons |
| [evaluation.md](evaluation.md) | EvalRunner, golden dataset, scoring, consistency checking |
| [configuration.md](configuration.md) | Environment variables, Settings, backend selection, secrets |

## Quick Reference

### Files by Layer

**Application Entry**
- `orchestrator/src/orchestrator/__main__.py` — CLI, component factory, scheduler startup

**Core Models**
- `orchestrator/src/orchestrator/models.py` — Domain models (TradeProposal, SentimentReport, etc.)
- `orchestrator/src/orchestrator/config.py` — Pydantic Settings

**Pipeline**
- `orchestrator/src/orchestrator/pipeline/runner.py` — Main orchestration
- `orchestrator/src/orchestrator/pipeline/scheduler.py` — APScheduler wrapper
- `orchestrator/src/orchestrator/pipeline/aggregator.py` — Multi-agent validation

**Agents**
- `orchestrator/src/orchestrator/agents/base.py` — BaseAgent[T] generic wrapper
- `orchestrator/src/orchestrator/agents/sentiment.py` — SentimentAgent
- `orchestrator/src/orchestrator/agents/market.py` — MarketAgent
- `orchestrator/src/orchestrator/agents/proposer.py` — ProposerAgent

**Skills** (External prompts)
- `.claude/skills/sentiment/SKILL.md` — Sentiment analysis methodology
- `.claude/skills/market/SKILL.md` — Technical analysis methodology
- `.claude/skills/proposer/SKILL.md` — Trade proposal generation logic

**LLM Integration**
- `orchestrator/src/orchestrator/llm/client.py` — LLMClient (async wrapper)
- `orchestrator/src/orchestrator/llm/backend.py` — LiteLLMBackend, ClaudeCLIBackend
- `orchestrator/src/orchestrator/llm/schema_validator.py` — JSON extraction & validation

**Exchange**
- `orchestrator/src/orchestrator/exchange/client.py` — ExchangeClient (CCXT wrapper)
- `orchestrator/src/orchestrator/exchange/data_fetcher.py` — DataFetcher (snapshots)
- `orchestrator/src/orchestrator/exchange/paper_engine.py` — Paper trading simulation
- `orchestrator/src/orchestrator/exchange/price_monitor.py` — SL/TP tracking

**Risk Management**
- `orchestrator/src/orchestrator/risk/checker.py` — RiskChecker (validation rules)
- `orchestrator/src/orchestrator/risk/position_sizer.py` — RiskPercentSizer, MarginSizer

**Execution**
- `orchestrator/src/orchestrator/execution/executor.py` — OrderExecutor interface, Paper/Live implementations

**Approval & Telegram**
- `orchestrator/src/orchestrator/approval/manager.py` — ApprovalManager state machine
- `orchestrator/src/orchestrator/telegram/bot.py` — SentinelBot handlers
- `orchestrator/src/orchestrator/telegram/formatters.py` — Message formatting

**Storage**
- `orchestrator/src/orchestrator/storage/models.py` — SQLModel tables
- `orchestrator/src/orchestrator/storage/database.py` — SQLAlchemy setup
- `orchestrator/src/orchestrator/storage/repository.py` — Repository pattern implementations

**Statistics**
- `orchestrator/src/orchestrator/stats/calculator.py` — PnL, win rate, Sharpe, drawdown

**Evaluation**
- `orchestrator/src/orchestrator/eval/runner.py` — EvalRunner
- `orchestrator/src/orchestrator/eval/dataset.py` — Dataset loading
- `orchestrator/src/orchestrator/eval/scorers.py` — RuleScorer

**Logging**
- `orchestrator/src/orchestrator/logging.py` — structlog setup

## Key Concepts

### Immutability
All Pydantic models use `frozen=True`. Updates via `model_copy(update={...})`.

### Async-First
LLM calls and exchange data fetches use `asyncio`.

### Repository Pattern
Data access through repository classes, not direct DB queries.

### Skill-Based Prompts
Instead of embedded prompts, agents reference external skill files in `.claude/skills/`.

### Structured Logging
All logs include `run_id` context for pipeline tracing.

## Testing

Run tests:
```bash
cd orchestrator
uv run pytest -v --cov=orchestrator
```

Test files:
- `tests/unit/test_*.py` — Unit tests for each module
- `tests/integration/` — End-to-end tests (if enabled)

Key test fixtures:
- In-memory SQLite (`:memory:`)
- Mocked CCXT exchange
- Mocked Telegram API
- Fixed seeds for deterministic LLM outputs

## Development Workflow

### 1. Understanding the Codebase
- Start with [INDEX.md](INDEX.md) for overview
- Read relevant codemap for specific domain
- Follow import paths to find implementations

### 2. Making Changes
- Keep immutability (frozen Pydantic models)
- Use repository pattern for data access
- Bind `run_id` to structlog context
- Add tests in `tests/unit/`

### 3. Testing Changes
```bash
# Run tests
uv run pytest -v

# Test specific module
uv run pytest tests/unit/test_risk_checker.py -v

# With coverage
uv run pytest --cov=orchestrator
```

### 4. Debugging
- Use structured logs: `uv run python -m orchestrator 2>&1 | grep "run_id=..."`
- Check database: `sqlite3 data/sentinel.db "SELECT * FROM pipeline_runs LIMIT 10;"`
- Telegram test messages via `/run BTC`

## Documentation Maintenance

These codemaps are **generated from code**, not manually maintained. Keep them fresh:

1. When adding new modules, document in appropriate codemap
2. When changing architecture, update [INDEX.md](INDEX.md)
3. When adding skills, document in [agents-skills.md](agents-skills.md)
4. Run `uv run pytest` to verify all imports/code paths still exist

## Related Documents

See project root:
- **README.md** — User-facing guide, CLI commands, quick start
- **CLAUDE.md** — Development conventions (immutability, async-first, etc.)
- **docs/plans/** — Design documents for major features
- **.claude/skills/** — Agent prompt templates

## Version Info

- **Last Updated:** 2026-02-28
- **Coverage:** All core modules + pipeline + skills-first architecture
- **Status:** Production-ready documentation

## Questions?

- Architecture questions → Read INDEX.md + relevant codemap
- Code questions → Search for filename in codemaps, follow to source
- Git history → `git log --all --grep="<feature>"` with commit refs in plans/

---

**Remember:** These codemaps are your single source of truth for understanding where features live in the code. Keep them in sync with reality.
