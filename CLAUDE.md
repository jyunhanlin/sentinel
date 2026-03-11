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
