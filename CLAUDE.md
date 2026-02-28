# Sentinel Orchestrator

## Project Structure

- `orchestrator/` — Python project (uv managed)
- `executor/` — Rust project (future)
- `schemas/` — Cross-language JSON Schema definitions

## Development

```bash
cd orchestrator
uv sync --all-extras
uv run pytest -v --cov=orchestrator
uv run ruff check src/ tests/
```

## Architecture

See `.claude/plans/2026-02-18-sentinel-design.md`

## Conventions

- Immutable models: all Pydantic models use `frozen=True`
- Async-first: exchange and LLM calls use asyncio
- Structured logging: use structlog, always bind run_id for pipeline context
- Repository pattern: data access through repository classes, not direct DB queries
- Risk % position sizing with strategy pattern

## Trading Skills

When running pipeline analysis via `claude -p`, the following skills are available:
- sentiment: `.claude/skills/sentiment/SKILL.md` — market sentiment analysis
- market: `.claude/skills/market/SKILL.md` — technical analysis
- proposer: `.claude/skills/proposer/SKILL.md` — trade proposal generation

Each skill file contains the full analysis methodology, decision criteria, and output schema.
Agents reference skills by name in their prompts; Claude reads the SKILL.md via the Read tool.
