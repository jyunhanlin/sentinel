# Sentinel Orchestrator

## Development

```bash
cd orchestrator
uv sync --all-extras
uv run pytest -v --cov=orchestrator
uv run ruff check src/ tests/
```

## Worktrees

Git worktrees are stored in `.worktrees/` at the project root.

## Conventions

- Immutable models: all Pydantic models use `frozen=True`
- Async-first: exchange and LLM calls use asyncio
- Structured logging: use structlog, always bind run_id for pipeline context
- Repository pattern: data access through repository classes, not direct DB queries
- Risk % position sizing with strategy pattern
- Schema validation: LLM outputs validated against JSON schemas in `schemas/`

## Trading Skills

Pipeline analysis skills live in `.claude/skills/*/SKILL.md`. Each contains the full methodology, criteria, and output schema. Agents invoke skills via `claude -p`.

## Architecture

5 analysis agents run in parallel → Proposer synthesizes a TradeProposal → Critic validates → RefinementLoop revises if rejected (up to N rounds) → Telegram approval → paper/live execution.

See `docs/CODEMAPS/architecture.md` for detailed pipeline flow, callback wiring, and module map.

## Key Files

| File | Role |
|------|------|
| `models.py` | All domain models (TechnicalAnalysis, TradeProposal, CritiqueResult, etc.) |
| `config.py` | `Settings` class — all env vars via pydantic-settings |
| `agents/base.py` | `BaseAgent[T]` abstract base — override `_build_prompt()` and `_get_default_output()` |
| `pipeline/runner.py` | `PipelineRunner` — orchestrates the full pipeline per symbol |
| `pipeline/refinement.py` | `RefinementLoop` — proposer ↔ critic feedback loop |
| `storage/repository.py` | 6 repository classes wrapping all DB access |
| `telegram/bot.py` | `SentinelBot` — Telegram handlers + approval callbacks |
| `exchange/paper_engine.py` | `PaperEngine` — simulated order fills + position tracking |

## Database

| Table | Repository | Purpose |
|-------|-----------|---------|
| `pipeline_runs` | `PipelineRepository` | Pipeline execution lifecycle |
| `llm_calls` | `LLMCallRepository` | LLM audit trail (model, tokens, cost) |
| `trade_proposals` | `TradeProposalRepository` | Full proposal JSON storage |
| `paper_trades` | `PaperTradeRepository` | Open/closed positions + PnL |
| `approval_records` | `ApprovalRepository` | Approval state machine |
| `account_snapshots` | `AccountSnapshotRepository` | Equity, Sharpe, drawdown snapshots |

## Adding a New Agent

1. Create `agents/<name>.py` — subclass `BaseAgent[YourOutputModel]`
2. Set `output_model`, `_skill_name`, implement `_build_prompt()` and `_get_default_output()`
3. Create the skill at `.claude/skills/<name>/SKILL.md` with methodology + output schema
4. Add the agent call to `pipeline/runner.py` (parallel or serial depending on dependencies)
5. Add JSON schema to `schemas/` if the output needs cross-language validation
