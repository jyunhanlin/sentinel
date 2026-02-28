# M0: Project Skeleton Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up the project skeleton with config, CCXT data fetching, Telegram bot basic commands, and structured logging — the foundation for all subsequent milestones.

**Architecture:** Monorepo with `orchestrator/` (Python, uv) and `executor/` (Rust, future). Python package uses src layout at `orchestrator/src/orchestrator/`. All config via environment variables with Pydantic Settings. Async-first with asyncio.

**Tech Stack:** Python 3.12+, uv, Pydantic v2, CCXT (async), python-telegram-bot, structlog, SQLModel, pytest, pytest-asyncio

**Design doc:** `.claude/plans/2026-02-18-sentinel-orchestrator-design.md`

---

### Task 1: Initialize uv Project & Directory Structure

**Files:**
- Create: `orchestrator/pyproject.toml`
- Create: `orchestrator/src/orchestrator/__init__.py`
- Create: `orchestrator/src/orchestrator/agents/__init__.py`
- Create: `orchestrator/src/orchestrator/pipeline/__init__.py`
- Create: `orchestrator/src/orchestrator/exchange/__init__.py`
- Create: `orchestrator/src/orchestrator/risk/__init__.py`
- Create: `orchestrator/src/orchestrator/telegram/__init__.py`
- Create: `orchestrator/src/orchestrator/storage/__init__.py`
- Create: `orchestrator/src/orchestrator/llm/__init__.py`
- Create: `schemas/.gitkeep`
- Create: `.env.example`

**Step 1: Create project structure with uv**

```bash
cd orchestrator
uv init --lib --name orchestrator
```

If uv init doesn't produce the exact layout we want, manually adjust.

**Step 2: Create pyproject.toml**

```toml
[project]
name = "orchestrator"
version = "0.1.0"
description = "Multi-LLM trade proposal orchestrator"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "structlog>=24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "ruff>=0.8",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.backends"

[tool.hatch.build.targets.wheel]
packages = ["src/orchestrator"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

**Step 3: Create all `__init__.py` files for subpackages**

Each `__init__.py` is empty for now. Create directories:
- `orchestrator/src/orchestrator/`
- `orchestrator/src/orchestrator/agents/`
- `orchestrator/src/orchestrator/pipeline/`
- `orchestrator/src/orchestrator/exchange/`
- `orchestrator/src/orchestrator/risk/`
- `orchestrator/src/orchestrator/telegram/`
- `orchestrator/src/orchestrator/storage/`
- `orchestrator/src/orchestrator/llm/`
- `orchestrator/tests/unit/`
- `orchestrator/tests/integration/`
- `schemas/`

**Step 4: Create .env.example**

```env
# Telegram
TELEGRAM_BOT_TOKEN=your-bot-token-here
TELEGRAM_ADMIN_CHAT_IDS=123456789

# Exchange
EXCHANGE_ID=binance
EXCHANGE_API_KEY=
EXCHANGE_API_SECRET=

# LLM
ANTHROPIC_API_KEY=your-anthropic-key-here

# Database
DATABASE_URL=sqlite:///data/sentinel.db

# Pipeline
PIPELINE_INTERVAL_MINUTES=15
PIPELINE_SYMBOLS=BTC/USDT:USDT,ETH/USDT:USDT

# Risk
MAX_SINGLE_RISK_PCT=2.0
MAX_TOTAL_EXPOSURE_PCT=20.0
MAX_DAILY_LOSS_PCT=5.0
```

**Step 5: Install dependencies and verify**

```bash
cd orchestrator
uv sync --all-extras
uv run python -c "import orchestrator; print('OK')"
```

Expected: `OK`

**Step 6: Verify tests run (even with no tests yet)**

```bash
cd orchestrator
uv run pytest --co
```

Expected: "no tests ran" or similar

**Step 7: Commit**

```bash
git add orchestrator/ schemas/ .env.example
git commit -m "chore: initialize uv project with directory structure"
```

---

### Task 2: Configuration Management

**Files:**
- Create: `orchestrator/src/orchestrator/config.py`
- Test: `orchestrator/tests/unit/test_config.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_config.py
import pytest
from orchestrator.config import Settings


def test_settings_loads_defaults():
    settings = Settings(
        telegram_bot_token="test-token",
        telegram_admin_chat_ids=[123],
        anthropic_api_key="test-key",
    )
    assert settings.exchange_id == "binance"
    assert settings.pipeline_interval_minutes == 15
    assert settings.max_single_risk_pct == 2.0
    assert settings.database_url == "sqlite:///data/sentinel.db"


def test_settings_requires_telegram_token():
    with pytest.raises(Exception):
        Settings(
            telegram_admin_chat_ids=[123],
            anthropic_api_key="test-key",
        )


def test_settings_parses_symbols():
    settings = Settings(
        telegram_bot_token="test-token",
        telegram_admin_chat_ids=[123],
        anthropic_api_key="test-key",
        pipeline_symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
    )
    assert len(settings.pipeline_symbols) == 2
    assert settings.pipeline_symbols[0] == "BTC/USDT:USDT"
```

**Step 2: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.config'`

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/config.py
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "", "env_file": ".env", "env_file_encoding": "utf-8"}

    # Telegram
    telegram_bot_token: str
    telegram_admin_chat_ids: list[int]

    # Exchange
    exchange_id: str = "binance"
    exchange_api_key: str = ""
    exchange_api_secret: str = ""

    # LLM
    anthropic_api_key: str

    # Database
    database_url: str = "sqlite:///data/sentinel.db"

    # Pipeline
    pipeline_interval_minutes: int = 15
    pipeline_symbols: list[str] = Field(default=["BTC/USDT:USDT", "ETH/USDT:USDT"])

    # Risk
    max_single_risk_pct: float = 2.0
    max_total_exposure_pct: float = 20.0
    max_daily_loss_pct: float = 5.0
    max_consecutive_losses: int = 5
```

**Step 4: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_config.py -v
```

Expected: 3 passed

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/config.py orchestrator/tests/unit/test_config.py
git commit -m "feat: add configuration management with pydantic-settings"
```

---

### Task 3: Structured Logging Setup

**Files:**
- Create: `orchestrator/src/orchestrator/logging.py`
- Test: `orchestrator/tests/unit/test_logging.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_logging.py
import json
import structlog
from orchestrator.logging import setup_logging


def test_setup_logging_returns_logger():
    setup_logging(json_output=False)
    logger = structlog.get_logger("test")
    assert logger is not None


def test_logger_binds_context():
    setup_logging(json_output=False)
    logger = structlog.get_logger("test")
    bound = logger.bind(run_id="abc-123", symbol="BTC/USDT:USDT")
    # Should not raise
    assert bound is not None
```

**Step 2: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_logging.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/logging.py
import structlog


def setup_logging(*, json_output: bool = True) -> None:
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
```

**Step 4: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_logging.py -v
```

Expected: 2 passed

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/logging.py orchestrator/tests/unit/test_logging.py
git commit -m "feat: add structured logging with structlog"
```

---

### Task 4: Pydantic Domain Models (from schemas)

**Files:**
- Create: `orchestrator/src/orchestrator/models.py`
- Create: `schemas/trade_proposal.json`
- Create: `schemas/sentiment_report.json`
- Create: `schemas/market_interpretation.json`
- Test: `orchestrator/tests/unit/test_models.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_models.py
import uuid
import pytest
from pydantic import ValidationError
from orchestrator.models import (
    SentimentReport,
    KeyEvent,
    MarketInterpretation,
    KeyLevel,
    TradeProposal,
    EntryOrder,
    Side,
    Trend,
    VolatilityRegime,
)


class TestSentimentReport:
    def test_valid_report(self):
        report = SentimentReport(
            sentiment_score=72,
            key_events=[
                KeyEvent(event="BTC ETF inflows", impact="positive", source="Bloomberg")
            ],
            sources=["twitter", "news"],
            confidence=0.8,
        )
        assert report.sentiment_score == 72
        assert len(report.key_events) == 1

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            SentimentReport(
                sentiment_score=101,
                key_events=[],
                sources=[],
                confidence=0.5,
            )

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            SentimentReport(
                sentiment_score=50,
                key_events=[],
                sources=[],
                confidence=1.5,
            )


class TestMarketInterpretation:
    def test_valid_interpretation(self):
        interp = MarketInterpretation(
            trend=Trend.UP,
            volatility_regime=VolatilityRegime.MEDIUM,
            key_levels=[KeyLevel(type="support", price=93000.0)],
            risk_flags=["funding_elevated"],
        )
        assert interp.trend == Trend.UP
        assert len(interp.key_levels) == 1


class TestTradeProposal:
    def test_valid_proposal(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=1.5,
            stop_loss=93000.0,
            take_profit=[95500.0, 97000.0],
            time_horizon="4h",
            confidence=0.75,
            invalid_if=["funding_rate > 0.05%"],
            rationale="Bullish momentum",
        )
        assert proposal.proposal_id is not None
        assert proposal.side == Side.LONG

    def test_flat_side_no_stop_loss_required(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT",
            side=Side.FLAT,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=0.0,
            stop_loss=None,
            take_profit=[],
            time_horizon="4h",
            confidence=0.5,
            invalid_if=[],
            rationale="No trade",
        )
        assert proposal.side == Side.FLAT

    def test_risk_pct_negative_rejected(self):
        with pytest.raises(ValidationError):
            TradeProposal(
                symbol="BTC/USDT:USDT",
                side=Side.LONG,
                entry=EntryOrder(type="market"),
                position_size_risk_pct=-1.0,
                stop_loss=93000.0,
                take_profit=[],
                time_horizon="4h",
                confidence=0.75,
                invalid_if=[],
                rationale="test",
            )
```

**Step 2: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_models.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/models.py
from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


# --- Enums ---

class Side(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class Trend(StrEnum):
    UP = "up"
    DOWN = "down"
    RANGE = "range"


class VolatilityRegime(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# --- Sentiment ---

class KeyEvent(BaseModel, frozen=True):
    event: str
    impact: str
    source: str


class SentimentReport(BaseModel, frozen=True):
    sentiment_score: int = Field(ge=0, le=100)
    key_events: list[KeyEvent]
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


# --- Market ---

class KeyLevel(BaseModel, frozen=True):
    type: str
    price: float


class MarketInterpretation(BaseModel, frozen=True):
    trend: Trend
    volatility_regime: VolatilityRegime
    key_levels: list[KeyLevel]
    risk_flags: list[str]


# --- Trade Proposal ---

class EntryOrder(BaseModel, frozen=True):
    type: str  # "market" or "limit"
    price: float | None = None


class TradeProposal(BaseModel, frozen=True):
    proposal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    side: Side
    entry: EntryOrder
    position_size_risk_pct: float = Field(ge=0.0)
    stop_loss: float | None = None
    take_profit: list[float]
    time_horizon: str
    confidence: float = Field(ge=0.0, le=1.0)
    invalid_if: list[str]
    rationale: str
```

**Step 4: Also create the JSON Schema files**

```json
// schemas/sentiment_report.json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "SentimentReport",
  "type": "object",
  "required": ["sentiment_score", "key_events", "sources", "confidence"],
  "properties": {
    "sentiment_score": {"type": "integer", "minimum": 0, "maximum": 100},
    "key_events": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["event", "impact", "source"],
        "properties": {
          "event": {"type": "string"},
          "impact": {"type": "string"},
          "source": {"type": "string"}
        }
      }
    },
    "sources": {"type": "array", "items": {"type": "string"}},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
  }
}
```

```json
// schemas/market_interpretation.json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "MarketInterpretation",
  "type": "object",
  "required": ["trend", "volatility_regime", "key_levels", "risk_flags"],
  "properties": {
    "trend": {"type": "string", "enum": ["up", "down", "range"]},
    "volatility_regime": {"type": "string", "enum": ["low", "medium", "high"]},
    "key_levels": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["type", "price"],
        "properties": {
          "type": {"type": "string"},
          "price": {"type": "number"}
        }
      }
    },
    "risk_flags": {"type": "array", "items": {"type": "string"}}
  }
}
```

```json
// schemas/trade_proposal.json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "TradeProposal",
  "type": "object",
  "required": ["symbol", "side", "entry", "position_size_risk_pct", "take_profit", "time_horizon", "confidence", "invalid_if", "rationale"],
  "properties": {
    "proposal_id": {"type": "string"},
    "symbol": {"type": "string"},
    "side": {"type": "string", "enum": ["long", "short", "flat"]},
    "entry": {
      "type": "object",
      "required": ["type"],
      "properties": {
        "type": {"type": "string", "enum": ["market", "limit"]},
        "price": {"type": "number"}
      }
    },
    "position_size_risk_pct": {"type": "number", "minimum": 0},
    "stop_loss": {"type": ["number", "null"]},
    "take_profit": {"type": "array", "items": {"type": "number"}},
    "time_horizon": {"type": "string"},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "invalid_if": {"type": "array", "items": {"type": "string"}},
    "rationale": {"type": "string"}
  }
}
```

**Step 5: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_models.py -v
```

Expected: 6 passed

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/models.py orchestrator/tests/unit/test_models.py schemas/
git commit -m "feat: add core domain models with Pydantic and JSON schemas"
```

---

### Task 5: Storage Layer (SQLModel + SQLite)

**Files:**
- Create: `orchestrator/src/orchestrator/storage/models.py`
- Create: `orchestrator/src/orchestrator/storage/database.py`
- Create: `orchestrator/src/orchestrator/storage/repository.py`
- Test: `orchestrator/tests/unit/test_storage.py`

**Step 1: Add sqlmodel dependency**

```bash
cd orchestrator && uv add sqlmodel aiosqlite
```

**Step 2: Write the failing test**

```python
# orchestrator/tests/unit/test_storage.py
import pytest
from sqlmodel import Session, create_engine, SQLModel
from orchestrator.storage.models import PipelineRunRecord, LLMCallRecord
from orchestrator.storage.database import create_db_engine, init_db
from orchestrator.storage.repository import PipelineRepository


@pytest.fixture
def engine():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    with Session(engine) as session:
        yield session


class TestPipelineRunRecord:
    def test_create_run(self, session):
        run = PipelineRunRecord(
            run_id="test-run-001",
            symbol="BTC/USDT:USDT",
            status="running",
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        assert run.id is not None
        assert run.run_id == "test-run-001"


class TestPipelineRepository:
    def test_save_and_get_run(self, session):
        repo = PipelineRepository(session)
        run = repo.create_run(run_id="test-001", symbol="BTC/USDT:USDT")
        fetched = repo.get_run("test-001")
        assert fetched is not None
        assert fetched.run_id == "test-001"
        assert fetched.status == "running"

    def test_update_run_status(self, session):
        repo = PipelineRepository(session)
        repo.create_run(run_id="test-002", symbol="BTC/USDT:USDT")
        updated = repo.update_run_status("test-002", "completed")
        assert updated.status == "completed"

    def test_get_nonexistent_run(self, session):
        repo = PipelineRepository(session)
        result = repo.get_run("nonexistent")
        assert result is None
```

**Step 3: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_storage.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 4: Write implementation**

```python
# orchestrator/src/orchestrator/storage/models.py
from __future__ import annotations

from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


class PipelineRunRecord(SQLModel, table=True):
    __tablename__ = "pipeline_runs"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(unique=True, index=True)
    symbol: str
    status: str = "running"  # running, completed, failed
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LLMCallRecord(SQLModel, table=True):
    __tablename__ = "llm_calls"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    agent_type: str  # sentiment, market, proposer
    prompt: str
    response: str
    model: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TradeProposalRecord(SQLModel, table=True):
    __tablename__ = "trade_proposals"

    id: int | None = Field(default=None, primary_key=True)
    proposal_id: str = Field(unique=True, index=True)
    run_id: str = Field(index=True)
    proposal_json: str  # Full JSON
    risk_check_result: str = ""  # approved / rejected
    risk_check_reason: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PaperTradeRecord(SQLModel, table=True):
    __tablename__ = "paper_trades"

    id: int | None = Field(default=None, primary_key=True)
    trade_id: str = Field(unique=True, index=True)
    proposal_id: str = Field(index=True)
    symbol: str
    side: str
    entry_price: float
    exit_price: float | None = None
    quantity: float
    pnl: float = 0.0
    fees: float = 0.0
    status: str = "open"  # open, closed
    opened_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: datetime | None = None


class AccountSnapshotRecord(SQLModel, table=True):
    __tablename__ = "account_snapshots"

    id: int | None = Field(default=None, primary_key=True)
    equity: float
    open_positions_count: int = 0
    daily_pnl: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

```python
# orchestrator/src/orchestrator/storage/database.py
from sqlmodel import SQLModel, create_engine, Session


def create_db_engine(database_url: str):
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(database_url, connect_args=connect_args)


def init_db(engine) -> None:
    SQLModel.metadata.create_all(engine)


def get_session(engine) -> Session:
    return Session(engine)
```

```python
# orchestrator/src/orchestrator/storage/repository.py
from __future__ import annotations

from sqlmodel import Session, select

from orchestrator.storage.models import PipelineRunRecord


class PipelineRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_run(self, *, run_id: str, symbol: str) -> PipelineRunRecord:
        run = PipelineRunRecord(run_id=run_id, symbol=symbol, status="running")
        self._session.add(run)
        self._session.commit()
        self._session.refresh(run)
        return run

    def get_run(self, run_id: str) -> PipelineRunRecord | None:
        statement = select(PipelineRunRecord).where(PipelineRunRecord.run_id == run_id)
        return self._session.exec(statement).first()

    def update_run_status(self, run_id: str, status: str) -> PipelineRunRecord:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        run.status = status
        self._session.add(run)
        self._session.commit()
        self._session.refresh(run)
        return run
```

**Step 5: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_storage.py -v
```

Expected: 4 passed

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/storage/ orchestrator/tests/unit/test_storage.py
git commit -m "feat: add storage layer with SQLModel and repository pattern"
```

---

### Task 6: CCXT Exchange Client

**Files:**
- Create: `orchestrator/src/orchestrator/exchange/client.py`
- Create: `orchestrator/src/orchestrator/exchange/data_fetcher.py`
- Test: `orchestrator/tests/unit/test_exchange.py`

**Step 1: Add ccxt dependency**

```bash
cd orchestrator && uv add ccxt
```

**Step 2: Write the failing test**

```python
# orchestrator/tests/unit/test_exchange.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from orchestrator.exchange.client import ExchangeClient
from orchestrator.exchange.data_fetcher import DataFetcher, MarketSnapshot


class TestExchangeClient:
    @pytest.mark.asyncio
    async def test_create_client(self):
        client = ExchangeClient(exchange_id="binance")
        assert client.exchange_id == "binance"

    @pytest.mark.asyncio
    async def test_fetch_ohlcv(self):
        client = ExchangeClient(exchange_id="binance")
        # Mock the internal ccxt exchange
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.return_value = [
            [1700000000000, 95000.0, 95500.0, 94500.0, 95200.0, 1000.0],
            [1700000060000, 95200.0, 95800.0, 95100.0, 95600.0, 800.0],
        ]
        client._exchange = mock_exchange

        candles = await client.fetch_ohlcv("BTC/USDT:USDT", "1h", limit=2)
        assert len(candles) == 2
        assert candles[0][1] == 95000.0  # open


class TestDataFetcher:
    @pytest.mark.asyncio
    async def test_fetch_snapshot(self):
        mock_client = AsyncMock(spec=ExchangeClient)
        mock_client.fetch_ohlcv.return_value = [
            [1700000000000, 95000.0, 95500.0, 94500.0, 95200.0, 1000.0],
        ]
        mock_client.fetch_funding_rate.return_value = 0.0001
        mock_client.fetch_ticker.return_value = {
            "last": 95200.0,
            "quoteVolume": 1000000.0,
        }

        fetcher = DataFetcher(mock_client)
        snapshot = await fetcher.fetch_snapshot("BTC/USDT:USDT", timeframe="1h")

        assert snapshot.symbol == "BTC/USDT:USDT"
        assert snapshot.current_price == 95200.0
        assert snapshot.funding_rate == 0.0001
        assert len(snapshot.ohlcv) == 1


class TestMarketSnapshot:
    def test_snapshot_is_immutable(self):
        snapshot = MarketSnapshot(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            current_price=95200.0,
            volume_24h=1000000.0,
            funding_rate=0.0001,
            ohlcv=[[1700000000000, 95000.0, 95500.0, 94500.0, 95200.0, 1000.0]],
        )
        assert snapshot.symbol == "BTC/USDT:USDT"
        with pytest.raises(Exception):
            snapshot.symbol = "ETH"  # type: ignore
```

**Step 3: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_exchange.py -v
```

Expected: FAIL

**Step 4: Write implementation**

```python
# orchestrator/src/orchestrator/exchange/client.py
from __future__ import annotations

import ccxt.async_support as ccxt


class ExchangeClient:
    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "",
        api_secret: str = "",
    ) -> None:
        self.exchange_id = exchange_id
        exchange_class = getattr(ccxt, exchange_id)
        self._exchange = exchange_class(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "options": {"defaultType": "swap"},
            }
        )

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", *, limit: int = 100
    ) -> list[list]:
        return await self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    async def fetch_funding_rate(self, symbol: str) -> float:
        result = await self._exchange.fetch_funding_rate(symbol)
        return result.get("fundingRate", 0.0)

    async def fetch_ticker(self, symbol: str) -> dict:
        return await self._exchange.fetch_ticker(symbol)

    async def close(self) -> None:
        await self._exchange.close()
```

```python
# orchestrator/src/orchestrator/exchange/data_fetcher.py
from __future__ import annotations

from pydantic import BaseModel

from orchestrator.exchange.client import ExchangeClient


class MarketSnapshot(BaseModel, frozen=True):
    symbol: str
    timeframe: str
    current_price: float
    volume_24h: float
    funding_rate: float
    ohlcv: list[list]


class DataFetcher:
    def __init__(self, client: ExchangeClient) -> None:
        self._client = client

    async def fetch_snapshot(
        self, symbol: str, *, timeframe: str = "1h", limit: int = 100
    ) -> MarketSnapshot:
        ohlcv, funding, ticker = await self._parallel_fetch(symbol, timeframe, limit)

        return MarketSnapshot(
            symbol=symbol,
            timeframe=timeframe,
            current_price=ticker.get("last", 0.0),
            volume_24h=ticker.get("quoteVolume", 0.0),
            funding_rate=funding,
            ohlcv=ohlcv,
        )

    async def _parallel_fetch(
        self, symbol: str, timeframe: str, limit: int
    ) -> tuple[list[list], float, dict]:
        import asyncio

        ohlcv_task = self._client.fetch_ohlcv(symbol, timeframe, limit=limit)
        funding_task = self._client.fetch_funding_rate(symbol)
        ticker_task = self._client.fetch_ticker(symbol)

        ohlcv, funding, ticker = await asyncio.gather(
            ohlcv_task, funding_task, ticker_task
        )
        return ohlcv, funding, ticker
```

**Step 5: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_exchange.py -v
```

Expected: 4 passed

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/exchange/ orchestrator/tests/unit/test_exchange.py
git commit -m "feat: add CCXT exchange client and data fetcher"
```

---

### Task 7: Telegram Bot (Basic Commands)

**Files:**
- Create: `orchestrator/src/orchestrator/telegram/bot.py`
- Create: `orchestrator/src/orchestrator/telegram/formatters.py`
- Test: `orchestrator/tests/unit/test_telegram.py`

**Step 1: Add telegram dependency**

```bash
cd orchestrator && uv add python-telegram-bot
```

**Step 2: Write the failing test**

```python
# orchestrator/tests/unit/test_telegram.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from orchestrator.telegram.bot import SentinelBot, is_admin
from orchestrator.telegram.formatters import format_welcome, format_help


class TestIsAdmin:
    def test_admin_allowed(self):
        assert is_admin(123, admin_ids=[123, 456]) is True

    def test_non_admin_rejected(self):
        assert is_admin(789, admin_ids=[123, 456]) is False

    def test_empty_admin_list_rejects_all(self):
        assert is_admin(123, admin_ids=[]) is False


class TestFormatters:
    def test_format_welcome(self):
        msg = format_welcome()
        assert "Sentinel" in msg
        assert "/help" in msg

    def test_format_help(self):
        msg = format_help()
        assert "/status" in msg
        assert "/coin" in msg


class TestSentinelBot:
    def test_create_bot(self):
        bot = SentinelBot(token="test-token", admin_chat_ids=[123])
        assert bot.admin_chat_ids == [123]
```

**Step 3: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_telegram.py -v
```

Expected: FAIL

**Step 4: Write implementation**

```python
# orchestrator/src/orchestrator/telegram/formatters.py
def format_welcome() -> str:
    return (
        "Welcome to Sentinel Orchestrator!\n\n"
        "I analyze crypto markets using multiple AI models and generate "
        "trade proposals with risk management.\n\n"
        "Use /help to see available commands."
    )


def format_help() -> str:
    return (
        "Available commands:\n\n"
        "/start - Welcome message\n"
        "/status - Account overview & latest proposals\n"
        "/coin <symbol> - Detailed analysis for a symbol (e.g. /coin BTC)\n"
        "/history - Recent trade records\n"
        "/help - Show this message"
    )
```

```python
# orchestrator/src/orchestrator/telegram/bot.py
from __future__ import annotations

import structlog
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from orchestrator.telegram.formatters import format_welcome, format_help

logger = structlog.get_logger(__name__)


def is_admin(chat_id: int, *, admin_ids: list[int]) -> bool:
    return chat_id in admin_ids


class SentinelBot:
    def __init__(self, token: str, admin_chat_ids: list[int]) -> None:
        self.token = token
        self.admin_chat_ids = admin_chat_ids
        self._app: Application | None = None

    def build(self) -> Application:
        self._app = (
            Application.builder()
            .token(self.token)
            .build()
        )
        self._app.add_handler(CommandHandler("start", self._start_handler))
        self._app.add_handler(CommandHandler("help", self._help_handler))
        self._app.add_handler(CommandHandler("status", self._status_handler))
        self._app.add_handler(CommandHandler("coin", self._coin_handler))
        return self._app

    async def _check_admin(self, update: Update) -> bool:
        chat_id = update.effective_chat.id if update.effective_chat else 0
        if not is_admin(chat_id, admin_ids=self.admin_chat_ids):
            logger.warning("unauthorized_access", chat_id=chat_id)
            return False
        return True

    async def _start_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        await update.message.reply_text(format_welcome())

    async def _help_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        await update.message.reply_text(format_help())

    async def _status_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        # Placeholder — will be wired to pipeline in M1
        await update.message.reply_text("Status: Paper trading mode. No active proposals yet.")

    async def _coin_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /coin <symbol> (e.g. /coin BTC)")
            return
        symbol = args[0].upper()
        # Placeholder — will be wired to pipeline in M1
        await update.message.reply_text(f"Analysis for {symbol}: Coming soon in M1.")
```

**Step 5: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_telegram.py -v
```

Expected: 5 passed

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/ orchestrator/tests/unit/test_telegram.py
git commit -m "feat: add Telegram bot with basic commands and admin whitelist"
```

---

### Task 8: Application Entrypoint

**Files:**
- Create: `orchestrator/src/orchestrator/__main__.py`
- Test: `orchestrator/tests/unit/test_main.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_main.py
from orchestrator.__main__ import create_app_components


def test_create_app_components():
    """Verify we can construct core components without starting services."""
    components = create_app_components(
        telegram_bot_token="test-token",
        telegram_admin_chat_ids=[123],
        exchange_id="binance",
        database_url="sqlite:///:memory:",
        anthropic_api_key="test-key",
    )
    assert "bot" in components
    assert "exchange_client" in components
    assert "db_engine" in components
```

**Step 2: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_main.py -v
```

Expected: FAIL

**Step 3: Write implementation**

```python
# orchestrator/src/orchestrator/__main__.py
from __future__ import annotations

import asyncio

import structlog

from orchestrator.config import Settings
from orchestrator.exchange.client import ExchangeClient
from orchestrator.logging import setup_logging
from orchestrator.storage.database import create_db_engine, init_db
from orchestrator.telegram.bot import SentinelBot

logger = structlog.get_logger(__name__)


def create_app_components(
    *,
    telegram_bot_token: str,
    telegram_admin_chat_ids: list[int],
    exchange_id: str,
    database_url: str,
    anthropic_api_key: str,
) -> dict:
    db_engine = create_db_engine(database_url)
    init_db(db_engine)

    exchange_client = ExchangeClient(exchange_id=exchange_id)
    bot = SentinelBot(token=telegram_bot_token, admin_chat_ids=telegram_admin_chat_ids)

    return {
        "bot": bot,
        "exchange_client": exchange_client,
        "db_engine": db_engine,
    }


def main() -> None:
    setup_logging(json_output=True)

    settings = Settings()  # type: ignore[call-arg]
    logger.info("starting_sentinel", exchange=settings.exchange_id)

    components = create_app_components(
        telegram_bot_token=settings.telegram_bot_token,
        telegram_admin_chat_ids=settings.telegram_admin_chat_ids,
        exchange_id=settings.exchange_id,
        database_url=settings.database_url,
        anthropic_api_key=settings.anthropic_api_key,
    )

    app = components["bot"].build()
    logger.info("bot_ready", admin_ids=settings.telegram_admin_chat_ids)
    app.run_polling()


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_main.py -v
```

Expected: 1 passed

**Step 5: Run all tests to verify nothing is broken**

```bash
cd orchestrator && uv run pytest -v --tb=short
```

Expected: All tests pass (approximately 16-18 tests)

**Step 6: Check coverage**

```bash
cd orchestrator && uv run pytest --cov=orchestrator --cov-report=term-missing
```

Expected: 80%+ coverage

**Step 7: Commit**

```bash
git add orchestrator/src/orchestrator/__main__.py orchestrator/tests/unit/test_main.py
git commit -m "feat: add application entrypoint wiring all M0 components"
```

---

### Task 9: Final Verification & CLAUDE.md

**Files:**
- Create: `CLAUDE.md`

**Step 1: Run full test suite with coverage**

```bash
cd orchestrator && uv run pytest -v --cov=orchestrator --cov-report=term-missing
```

Expected: All pass, 80%+ coverage

**Step 2: Run linter**

```bash
cd orchestrator && uv run ruff check src/ tests/
```

Expected: No errors

**Step 3: Verify the app can be imported**

```bash
cd orchestrator && uv run python -c "from orchestrator.config import Settings; print('config OK')"
cd orchestrator && uv run python -c "from orchestrator.models import TradeProposal; print('models OK')"
cd orchestrator && uv run python -c "from orchestrator.exchange.client import ExchangeClient; print('exchange OK')"
```

Expected: All print OK

**Step 4: Create CLAUDE.md**

Create `CLAUDE.md` at project root with project-specific instructions for development:

```markdown
# Sentinel Orchestrator

## Project Structure

- `orchestrator/` — Python project (uv managed)
- `executor/` — Rust project (future)
- `schemas/` — Cross-language JSON Schema definitions

## Development

cd orchestrator
uv sync --all-extras
uv run pytest -v --cov=orchestrator
uv run ruff check src/ tests/

## Architecture

See `.claude/plans/2026-02-18-sentinel-orchestrator-design.md`

## Conventions

- Immutable models: all Pydantic models use `frozen=True`
- Async-first: exchange and LLM calls use asyncio
- Structured logging: use structlog, always bind run_id for pipeline context
- Repository pattern: data access through repository classes, not direct DB queries
- Risk % position sizing with strategy pattern
```

**Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "chore: add CLAUDE.md with project conventions"
```

---

## M1 Outline (Next Plan — to be detailed when M0 is done)

Tasks for M1:
1. LLM client abstraction (LiteLLM wrapper)
2. Schema validator (validate LLM JSON output against Pydantic models)
3. Sentiment agent (LLM-1)
4. Market interpreter agent (LLM-2)
5. Trade proposer agent (LLM-3)
6. Aggregator (merge 3 outputs)
7. Pipeline runner (run_id, orchestrate agents)
8. Pipeline scheduler (APScheduler)
9. Wire /status and /coin to real pipeline data
10. TG push on new proposals

## M2 Outline (After M1)

Tasks for M2:
1. Position sizer (Risk % calculator)
2. Risk checker (rule engine)
3. Paper trading engine (account ledger, fill model)
4. Paper trade recording and PnL tracking
5. Wire pipeline → risk check → paper trade → TG report
6. /history command with real trade data
7. Account snapshot and drawdown tracking
