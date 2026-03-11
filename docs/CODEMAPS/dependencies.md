<!-- Generated: 2026-03-11 | Files scanned: 49 | Token estimate: ~500 -->

# Dependencies — External Services & Libraries

## Python Dependencies (pyproject.toml)

### Core

| Package | Purpose |
|---------|---------|
| pydantic / pydantic-settings | Immutable domain models, .env config |
| ccxt | Exchange API (Binance perpetual futures) |
| litellm | Multi-provider LLM API abstraction |
| python-telegram-bot | Telegram bot framework |
| sqlmodel + aiosqlite | SQLite ORM (async) |
| apscheduler | Cron-like job scheduling |
| aiohttp | HTTP client for external data APIs |
| structlog | Structured logging |

### Dev

| Package | Purpose |
|---------|---------|
| pytest / pytest-asyncio / pytest-cov | Testing |
| ruff | Linting |

## External Services

| Service | Used By | Purpose |
|---------|---------|---------|
| Binance (ccxt) | ExchangeClient | OHLCV, funding, OI, L/S ratios, order execution |
| Yahoo Finance | ExternalDataFetcher | DXY (DX-Y.NYB), S&P 500 (^GSPC) |
| CoinGecko | ExternalDataFetcher | BTC dominance via /api/v3/global |
| Telegram Bot API | SentinelBot | User interface, notifications, approval flow |
| Claude CLI | ClaudeCLIBackend | LLM calls via `claude -p` subprocess |
| LiteLLM providers | LiteLLMBackend | LLM calls via API (Anthropic, OpenAI, etc.) |

## Internal Dependency Graph

```
__main__.py (composition root)
  │
  ├── config ──────────── pydantic-settings
  ├── storage ─────────── sqlmodel, aiosqlite
  ├── llm ─────────────── litellm | claude CLI subprocess
  ├── agents ──────────── llm (via LLMClient)
  ├── exchange ────────── ccxt (async), aiohttp
  ├── approval ────────── storage (repositories)
  ├── execution ───────── exchange (ExchangeClient) | paper_engine
  ├── pipeline ────────── agents + exchange + approval + storage + execution
  ├── stats ───────────── storage (repositories, pure logic)
  └── telegram ────────── python-telegram-bot + llm (translations)
```

## File Structure

```
sentinel/
├── orchestrator/           Python project (uv managed)
│   ├── pyproject.toml
│   └── src/orchestrator/   Main package (49 files, 8,067 lines)
├── schemas/                Cross-language JSON schemas
│   └── trade_proposal.json
├── .claude/skills/         Agent skill definitions (6 internal SKILL.md files)
├── .agents/skills/         External skills (Binance Skills Hub, 7 skills)
├── skills-lock.json        External skill version locks
├── docs/plans/             Design & implementation docs (30+ files)
└── docs/CODEMAPS/          This documentation
```
