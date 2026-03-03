# Sentinel

Multi-LLM crypto trade proposal system with semi-auto execution via Telegram.

Five AI agents analyze technical structure (short & long term), positioning, catalysts, and cross-market correlations, then a Proposer agent synthesizes a trade proposal. Proposals pass through risk checks and require manual approval via Telegram inline keyboard before execution (paper or live).

## Architecture

```
Scheduler (every N min)
    │
    ├─ Technical (short-term) ──┐
    ├─ Technical (long-term) ───┤
    ├─ Positioning ─────────────┼─ Proposer → Aggregator → Risk Check
    ├─ Catalyst ────────────────┤                              │
    ├─ Correlation ─────────────┘                              ▼
    │                                           TG Push [Approve / Reject]
    │                                                          │
    │                                                 OrderExecutor (Paper/Live)
    │                                                          │
    └─ SL/TP Monitor ─────────────────────────────── Close Report
```

## Prerequisites

| Service            | Required       | How to get                                                        |
| ------------------ | -------------- | ----------------------------------------------------------------- |
| Anthropic API key  | Yes            | [console.anthropic.com](https://console.anthropic.com) → API Keys |
| Telegram Bot token | Yes            | TG → @BotFather → `/newbot`                                       |
| Telegram Chat ID   | Yes            | TG → @userinfobot → send any message                              |
| Binance API key    | Live mode only | [binance.com](https://www.binance.com) → API Management           |

## Quick Start

```bash
# Clone and setup
cd orchestrator
cp ../.env.example ../.env
# Edit ../.env — fill in ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_IDS

# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync --all-extras

# Run the bot (paper mode by default)
uv run python -m orchestrator
```

## Configuration

All config via environment variables or `.env` file at the repo root.

### Required

| Variable                  | Description                                |
| ------------------------- | ------------------------------------------ |
| `ANTHROPIC_API_KEY`       | Anthropic API key (required for `api` backend) |
| `TELEGRAM_BOT_TOKEN`      | Telegram bot token from @BotFather         |
| `TELEGRAM_ADMIN_CHAT_IDS` | Comma-separated chat IDs (admin whitelist) |

### Optional

| Variable                    | Default                       | Description                                           |
| --------------------------- | ----------------------------- | ----------------------------------------------------- |
| `EXCHANGE_ID`               | `binance`                     | CCXT exchange ID                                      |
| `EXCHANGE_API_KEY`          | (empty)                       | Exchange API key (live mode only)                     |
| `EXCHANGE_API_SECRET`       | (empty)                       | Exchange API secret (live mode only)                  |
| `LLM_BACKEND`               | `cli`                         | LLM backend: `api` (LiteLLM) or `cli` (Claude CLI)   |
| `LLM_MODEL`                 | `anthropic/claude-sonnet-4-6` | Default LLM model                                     |
| `LLM_MODEL_PREMIUM`         | `anthropic/claude-opus-4-6`   | Premium model for `/run` override                     |
| `CLAUDE_CLI_TIMEOUT`         | `600`                         | CLI subprocess timeout in seconds                     |
| `DATABASE_URL`              | `sqlite:///data/sentinel.db`  | SQLite database path                                  |
| `PIPELINE_INTERVAL_MINUTES` | `15`                          | Auto-run interval                                     |
| `PIPELINE_SYMBOLS`          | `BTC/USDT:USDT,ETH/USDT:USDT` | Symbols to analyze                                    |
| `TRADING_MODE`              | `paper`                       | `paper` or `live`                                     |
| `APPROVAL_TIMEOUT_MINUTES`  | `15`                          | Approval expiry time                                  |
| `PRICE_DEVIATION_THRESHOLD` | `0.01`                        | Max price change (1%) before rejecting stale approval |
| `PAPER_INITIAL_EQUITY`      | `10000.0`                     | Starting paper balance                                |
| `MAX_SINGLE_RISK_PCT`       | `2.0`                         | Max risk per trade                                    |
| `MAX_TOTAL_EXPOSURE_PCT`    | `20.0`                        | Max total open exposure                               |
| `MAX_DAILY_LOSS_PCT`        | `5.0`                         | Daily loss limit (pauses trading)                     |

## Telegram Commands

| Command                        | Description                                          |
| ------------------------------ | ---------------------------------------------------- |
| `/start`                       | Welcome message                                      |
| `/status`                      | Account overview and latest proposals                |
| `/coin <symbol>`               | Detailed analysis for a symbol (e.g. `/coin BTC`)    |
| `/run [symbol] [sonnet\|opus]` | Trigger pipeline manually                            |
| `/history`                     | Recent closed trades                                 |
| `/perf`                        | Performance report (PnL, win rate, Sharpe, drawdown) |
| `/resume`                      | Un-pause pipeline after risk pause                   |

Trade proposals appear with **Approve / Reject** inline buttons. Unanswered proposals expire after the configured timeout.

## CLI Subcommands

```bash
# Print performance report
uv run python -m orchestrator perf
```

## Development

```bash
cd orchestrator

# Run tests
uv run pytest -v --cov=orchestrator

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/
```

## Project Structure

```
sentinel/
├── orchestrator/              # Python project (uv)
│   └── src/orchestrator/
│       ├── agents/            # LLM agents (technical, positioning, catalyst, correlation, proposer)
│       ├── approval/          # Approval state machine + models
│       ├── exchange/          # CCXT client, data fetcher, external data (DXY, S&P, BTC.D), paper engine
│       ├── execution/         # OrderExecutor (paper/live)
│       ├── llm/               # LiteLLM / Claude CLI backend + schema validation
│       ├── pipeline/          # Runner, scheduler, aggregator
│       ├── risk/              # Risk checker + position sizer
│       ├── stats/             # Performance statistics calculator
│       ├── storage/           # SQLModel tables + repositories
│       ├── telegram/          # Bot handlers + formatters
│       ├── config.py          # Pydantic settings
│       └── models.py          # Core domain models
├── schemas/                   # Cross-language JSON Schema definitions
└── .env.example               # Environment variable template
```

## Tech Stack

| Layer           | Technology                                               |
| --------------- | -------------------------------------------------------- |
| Language        | Python 3.12+                                             |
| Package manager | [uv](https://docs.astral.sh/uv/)                         |
| LLM             | Claude via [LiteLLM](https://github.com/BerriAI/litellm) or Claude CLI |
| Exchange        | [CCXT](https://github.com/ccxt/ccxt) (async)             |
| Database        | SQLite + [SQLModel](https://sqlmodel.tiangolo.com/)      |
| Telegram        | [python-telegram-bot](https://python-telegram-bot.org/)  |
| Scheduler       | [APScheduler](https://apscheduler.readthedocs.io/)       |
| Logging         | [structlog](https://www.structlog.org/)                  |
| Validation      | [Pydantic v2](https://docs.pydantic.dev/)                |
