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

See `docs/plans/2026-02-18-sentinel-design.md`

## Worktrees

Git worktrees are stored in `.worktrees/` at the project root.

## Conventions

- Immutable models: all Pydantic models use `frozen=True`
- Async-first: exchange and LLM calls use asyncio
- Structured logging: use structlog, always bind run_id for pipeline context
- Repository pattern: data access through repository classes, not direct DB queries
- Risk % position sizing with strategy pattern

## Trading Skills

When running pipeline analysis via `claude -p`, the following skills are available:
- technical: `.claude/skills/technical/SKILL.md` — short/long-term technical analysis
- positioning: `.claude/skills/positioning/SKILL.md` — futures positioning & order flow
- catalyst: `.claude/skills/catalyst/SKILL.md` — event & news catalyst analysis
- correlation: `.claude/skills/correlation/SKILL.md` — cross-market correlation (DXY, S&P, BTC.D)
- proposer: `.claude/skills/proposer/SKILL.md` — trade proposal generation

Each skill file contains the full analysis methodology, decision criteria, and output schema.
Agents reference skills by name in their prompts; Claude reads the SKILL.md via the Read tool.
