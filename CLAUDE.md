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
