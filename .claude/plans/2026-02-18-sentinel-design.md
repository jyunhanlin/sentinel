# Sentinel Orchestrator — System Design

**Date:** 2026-02-18
**Status:** Approved

---

## Glossary

| Abbreviation | Full Name                      |
| ------------ | ------------------------------ |
| SL           | Stop Loss                      |
| TP           | Take Profit                    |
| OI           | Open Interest                  |
| PnL          | Profit and Loss                |
| OHLCV        | Open, High, Low, Close, Volume |
| ETF          | Exchange-Traded Fund           |
| HFT          | High Frequency Trading         |

---

## 1. Product Goal

Build a multi-LLM orchestration system that integrates market data (technical, order flow, sentiment, on-chain) to produce structured **trade proposals** (direction + position size + SL/TP + rationale), execute them via paper trading, and push results through Telegram.

### MVP Scope

- Analysis → Trade Proposal → Paper Trading → TG push
- Perpetual futures only (spot later)
- Single exchange: Binance (via CCXT abstraction)
- No live trading in MVP

### Out of MVP Scope

- Multi-exchange, multi-account, cross-exchange arbitrage, HFT
- Live trading (semi-auto comes in M4)
- Evaluation framework (M3)

---

## 2. Key Decisions

| Decision        | Choice                                | Rationale                                                                                 |
| --------------- | ------------------------------------- | ----------------------------------------------------------------------------------------- |
| Position sizing | Risk % (extensible to Notional %)     | Industry standard for quantitative risk control; strategy pattern allows future extension |
| Messaging       | Telegram                              | Free Bot API, unlimited messages, inline keyboards, crypto community standard             |
| Repo structure  | Monorepo                              | Single git repo, two independent projects (Python orchestrator + Rust executor)           |
| Trading target  | Perpetual futures first               | Richer data (funding, OI, liquidation), short-selling, leverage                           |
| Exchange        | CCXT → Binance                        | CCXT abstraction for future exchange swaps; Binance for liquidity                         |
| LLM provider    | Claude now, mixed later (via LiteLLM) | LiteLLM provides unified OpenAI-compatible interface for multi-provider routing           |
| Python tooling  | uv                                    | Fast, lockfile support, modern                                                            |
| Database        | SQLite (MVP)                          | Zero config, sufficient for single-user; ORM enables future PostgreSQL migration          |

---

## 3. Repository Structure

```
sentinel-orchestrator/
├── schemas/                            # Cross-language shared JSON Schema
│   ├── trade_proposal.json
│   ├── sentiment_report.json
│   └── market_interpretation.json
│
├── orchestrator/                       # Python project (uv)
│   ├── pyproject.toml
│   ├── src/
│   │   └── orchestrator/              # Python package
│   │       ├── __init__.py
│   │       ├── agents/                # LLM agents
│   │       │   ├── __init__.py
│   │       │   ├── sentiment.py       # LLM-1: news/sentiment analysis
│   │       │   ├── market.py          # LLM-2: technical/order flow interpretation
│   │       │   └── proposer.py        # LLM-3: trade proposal generation
│   │       ├── pipeline/              # Orchestration
│   │       │   ├── __init__.py
│   │       │   ├── scheduler.py       # Periodic trigger (APScheduler)
│   │       │   ├── runner.py          # Single pipeline execution
│   │       │   └── aggregator.py      # Merge 3 agent outputs → TradeProposal
│   │       ├── exchange/              # Market data & paper trading
│   │       │   ├── __init__.py
│   │       │   ├── client.py          # CCXT async wrapper
│   │       │   ├── data_fetcher.py    # OHLCV, funding, OI, etc.
│   │       │   └── paper_engine.py    # Simulated trading engine
│   │       ├── risk/                  # Risk management
│   │       │   ├── __init__.py
│   │       │   ├── checker.py         # Rule-based risk engine
│   │       │   └── position_sizer.py  # Risk % calculator (strategy pattern)
│   │       ├── telegram/              # Telegram bot
│   │       │   ├── __init__.py
│   │       │   ├── bot.py             # Bot setup & command handlers
│   │       │   └── formatters.py      # Proposal/status message formatting
│   │       ├── storage/               # Persistence
│   │       │   ├── __init__.py
│   │       │   ├── models.py          # SQLModel table definitions
│   │       │   └── repository.py      # Data access layer
│   │       ├── llm/                   # LLM abstraction
│   │       │   ├── __init__.py
│   │       │   ├── client.py          # LiteLLM wrapper
│   │       │   └── schema_validator.py # Validate LLM outputs against JSON Schema
│   │       └── config.py              # Configuration management
│   └── tests/
│       ├── unit/
│       └── integration/
│
├── executor/                           # Rust project (future)
│   ├── Cargo.toml                     # Cargo workspace
│   └── src/
│
├── docker-compose.yml
├── .env.example
├── CLAUDE.md
└── README.md
```

### Dependency Flow (within orchestrator)

```
config          ← standalone
storage         ← depends on config
llm             ← depends on config
exchange        ← depends on config, storage
risk            ← depends on config, storage
agents          ← depends on llm, config
pipeline        ← depends on agents, exchange, risk, storage
telegram        ← depends on pipeline, exchange, risk, storage (top-level integration)
```

---

## 4. Core Data Schemas

All LLM outputs MUST conform to schemas. Invalid outputs are rejected (retry or degrade).

### SentimentReport

```json
{
  "sentiment_score": 72,
  "key_events": [
    { "event": "BTC ETF inflows hit $1B", "impact": "positive", "source": "Bloomberg" }
  ],
  "sources": ["twitter", "news"],
  "confidence": 0.8
}
```

### MarketInterpretation

```json
{
  "trend": "up",
  "volatility_regime": "medium",
  "key_levels": [
    { "type": "support", "price": 93000 },
    { "type": "resistance", "price": 98000 }
  ],
  "risk_flags": ["funding_elevated", "oi_near_ath"]
}
```

### TradeProposal

```json
{
  "proposal_id": "uuid",
  "symbol": "BTC/USDT:USDT",
  "side": "long",
  "entry": { "type": "market" },
  "position_size_risk_pct": 1.5,
  "stop_loss": 93000,
  "take_profit": [95500, 97000],
  "time_horizon": "4h",
  "confidence": 0.75,
  "invalid_if": ["funding_rate > 0.05%"],
  "rationale": "Strong ETF inflows + breakout above key resistance with healthy funding"
}
```

---

## 5. Pipeline Data Flow

```
Scheduler (every X minutes)
    │
    ▼
Runner (generate run_id, fetch market snapshot)
    │
    ├──────────────┐
    ▼              ▼
LLM-1            LLM-2          ← parallel execution
(Sentiment)      (Market)
    │              │
    └──────┬───────┘
           ▼
         LLM-3                  ← depends on LLM-1 + LLM-2
       (Proposer)
           │
           ▼
       Aggregator               ← merge → validated TradeProposal
           │
           ▼
       Risk Check               ← hard gate (approve/reject)
           │
           ▼
    Paper Trading Engine        ← simulated execution
           │
           ▼
       TG Push                  ← notify user
```

### Step-by-step I/O

| Step         | Input                                  | Output                              |
| ------------ | -------------------------------------- | ----------------------------------- |
| Data Fetch   | symbol, timeframe                      | MarketSnapshot (OHLCV, funding, OI) |
| LLM-1        | News articles, social posts            | SentimentReport                     |
| LLM-2        | MarketSnapshot                         | MarketInterpretation                |
| LLM-3        | SentimentReport + MarketInterpretation | TradeProposal (raw)                 |
| Aggregator   | Three agent outputs                    | TradeProposal (validated)           |
| Risk Check   | TradeProposal + account state          | approved / rejected + reason        |
| Paper Engine | Approved TradeProposal                 | Simulated fill record               |
| TG Push      | Proposal + fill result                 | Message sent                        |

---

## 6. Risk Management

### Rules Engine (non-LLM, pure logic)

| Rule                   | Description                                  | Action         |
| ---------------------- | -------------------------------------------- | -------------- |
| Max single risk        | Single proposal risk% > threshold (e.g., 2%) | Reject         |
| Max total exposure     | All open positions total > threshold         | Reject new     |
| Direction sanity       | SL on wrong side of entry                    | Reject         |
| Max consecutive losses | N consecutive losses                         | Pause pipeline |
| Max daily loss         | Cumulative daily loss > threshold            | Pause pipeline |
| invalid_if check       | Proposal's self-declared cancel conditions   | Cancel         |

### Position Sizer (Risk % Mode)

```
position_size = (account_equity * risk_percent) / abs(entry - stop_loss)
```

Uses strategy pattern for future Notional % extension.

---

## 7. Paper Trading Engine

- **Account ledger:** Track equity, open positions, closed trades
- **Fill model (MVP simplified):**
  - Market order → immediate fill at current price ± slippage
  - Limit order → fills when next candle touches the price
  - SL/TP → same as limit order logic
- **Fees:** Configurable maker/taker rates (default 0.02%/0.05%)
- **Performance tracking:** PnL, win rate, profit factor, max drawdown, Sharpe ratio

---

## 8. Storage (SQLite + SQLModel)

| Table               | Purpose                                                               |
| ------------------- | --------------------------------------------------------------------- |
| `pipeline_runs`     | run_id, timestamp, symbol, status                                     |
| `llm_calls`         | run_id, agent_type, prompt, response, model, latency_ms, tokens, cost |
| `trade_proposals`   | proposal_id, run_id, full proposal JSON, risk_check_result            |
| `paper_trades`      | trade_id, proposal_id, entry/exit/pnl/fees                            |
| `account_snapshots` | timestamp, equity, open_positions, daily_pnl                          |

All pipeline inputs/outputs are stored for traceability and future evaluation.

---

## 9. Telegram Bot

### Commands

| Command         | Function                                       |
| --------------- | ---------------------------------------------- |
| `/start`        | Register + welcome                             |
| `/status`       | Account overview + latest proposals per symbol |
| `/coin BTC`     | Detailed breakdown for a single symbol         |
| `/history`      | Recent N trade records                         |
| `/approve <id>` | Confirm proposal (semi-auto mode, future)      |
| `/help`         | Command list                                   |

### Security

- Admin whitelist: only specified chat_id(s) can interact
- Configuration via environment variables

---

## 10. Technology Stack

| Layer             | Technology              | Rationale                           |
| ----------------- | ----------------------- | ----------------------------------- |
| Package manager   | uv                      | Fast, lockfile, modern              |
| Async             | asyncio (stdlib)        | LLM + exchange calls are I/O bound  |
| LLM calls         | LiteLLM                 | Unified multi-provider interface    |
| Schema validation | Pydantic v2             | Type safety, JSON Schema generation |
| Exchange API      | CCXT (async)            | Multi-exchange abstraction          |
| Database          | SQLite + SQLModel       | Zero config + ORM                   |
| Migrations        | Alembic                 | DB versioning                       |
| Telegram          | python-telegram-bot     | Mature, async support               |
| Scheduler         | APScheduler             | Lightweight, cron expressions       |
| Logging           | structlog               | Structured JSON logging             |
| Testing           | pytest + pytest-asyncio | Standard                            |

---

## 11. Milestones

- **M0 (1 week):** Project skeleton + config + CCXT data fetching + TG bot basic commands + structlog
- **M1 (2-3 weeks):** 3-model pipeline + schema validation + /status /coin + proposal push
- **M2 (2 weeks):** Paper trading engine + risk management + proposal→order→report loop
- **M3 (2-4 weeks):** Eval dataset + regression tests + monitoring dashboard
- **M4 (later):** Live semi-auto + Rust executor

---

## 12. Non-Functional Requirements

- **Observability:** Every pipeline run has a traceable run_id linking market snapshot, LLM outputs, and trade events
- **Reproducibility:** Fixed temperature/seed where supported; same input + model version → consistent output
- **Security:** API keys in env vars; TG admin whitelist; no secrets in code
- **Rate limiting & cost control:** Per-run token/cost ceiling; degrade to essential model only when exceeded
