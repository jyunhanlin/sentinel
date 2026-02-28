# Storage & Database Codemap

**Last Updated:** 2026-02-28

## Overview

All data persists to SQLite via SQLModel. Repository pattern encapsulates data access.

## Architecture

```
SQLite Database (data/sentinel.db)
    │
    ├─ SQLModel ORM
    │   └─ Tables defined in models.py
    │
    └─ Repository Layer
        ├─ PipelineRepository
        ├─ TradeProposalRepository
        ├─ PaperTradeRepository
        ├─ LLMCallRepository
        ├─ ApprovalRepository
        ├─ AccountSnapshotRepository
        └─ Others...
```

## Database Models

**File:** `orchestrator/src/orchestrator/storage/models.py`

All models use SQLModel (SQLAlchemy + Pydantic hybrid).

### Core Tables

#### PipelineRunRecord
```python
class PipelineRunRecord(SQLModel, table=True):
    __tablename__ = "pipeline_runs"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(unique=True, index=True)  # UUID for correlation
    symbol: str
    status: str = "running"  # "running", "completed", "failed"
    created_at: datetime
```

Tracks every pipeline execution for debugging and analytics.

#### LLMCallRecord
```python
class LLMCallRecord(SQLModel, table=True):
    __tablename__ = "llm_calls"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)  # Link to pipeline run
    agent_type: str  # "sentiment", "market", "proposer"
    prompt: str  # Full prompt sent to LLM
    response: str  # Full LLM response
    model: str  # e.g., "anthropic/claude-sonnet-4-6"
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    created_at: datetime
```

Records every LLM call for cost tracking and debugging.

#### TradeProposalRecord
```python
class TradeProposalRecord(SQLModel, table=True):
    __tablename__ = "trade_proposals"

    id: int | None = Field(default=None, primary_key=True)
    proposal_id: str = Field(unique=True, index=True)  # UUID
    run_id: str = Field(index=True)  # Link to pipeline run
    proposal_json: str  # Full JSON of TradeProposal model
    risk_check_result: str = ""  # "approved", "rejected"
    risk_check_reason: str = ""  # Why it was rejected
    created_at: datetime
```

Stores every proposal for audit trail.

#### PaperTradeRecord
```python
class PaperTradeRecord(SQLModel, table=True):
    __tablename__ = "paper_trades"

    id: int | None = Field(default=None, primary_key=True)
    trade_id: str = Field(unique=True, index=True)
    proposal_id: str = Field(index=True)
    symbol: str
    side: str  # "long", "short"
    entry_price: float
    exit_price: float | None = None
    quantity: float
    pnl: float = 0.0  # Profit/loss in USDT
    fees: float = 0.0
    risk_pct: float = 0.0  # Position risk as % of account
    status: str = "open"  # "open", "closed"
    mode: str = "paper"  # "paper" or "live"
    leverage: int = 1
    margin: float = 0.0  # Margin used
    liquidation_price: float = 0.0
    close_reason: str = ""  # "sl", "tp", "liquidation", "manual"
    stop_loss: float = 0.0
    take_profit_json: str = "[]"  # JSON-encoded list[float]
    opened_at: datetime
    closed_at: datetime | None = None
```

Records all executed trades (paper and live).

#### ApprovalRecord
```python
class ApprovalRecord(SQLModel, table=True):
    __tablename__ = "approval_records"

    id: int | None = Field(default=None, primary_key=True)
    approval_id: str = Field(unique=True, index=True)
    proposal_id: str = Field(index=True)
    run_id: str
    snapshot_price: float  # Price at approval time
    status: str = "pending"  # "pending", "approved", "rejected", "expired"
    message_id: int | None = None  # Telegram message ID
    created_at: datetime
    expires_at: datetime
    resolved_at: datetime | None = None
```

Tracks approval workflow for audit.

#### AccountSnapshotRecord
```python
class AccountSnapshotRecord(SQLModel, table=True):
    __tablename__ = "account_snapshots"

    id: int | None = Field(default=None, primary_key=True)
    equity: float
    open_positions_count: int
    daily_pnl: float
    total_pnl: float
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    created_at: datetime
```

Periodic snapshots of account metrics.

## Repository Pattern

**File:** `orchestrator/src/orchestrator/storage/repository.py`

Repositories encapsulate data access. All queries go through repositories, never direct DB access.

### Base Repository

```python
class BaseRepository[T]:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, obj: T) -> T:
        """Insert and return."""
        self._session.add(obj)
        self._session.commit()
        return obj

    def get(self, id: int) -> T | None:
        """Get by primary key."""
        return self._session.get(T, id)

    def all(self) -> list[T]:
        """Get all records."""
        return self._session.query(T).all()
```

### Specific Repositories

#### PipelineRepository
```python
class PipelineRepository(BaseRepository[PipelineRunRecord]):
    def get_by_run_id(self, run_id: str) -> PipelineRunRecord | None:
        return self._session.query(PipelineRunRecord).filter(
            PipelineRunRecord.run_id == run_id
        ).first()

    def get_latest_by_symbol(self, symbol: str, limit: int = 10) -> list[PipelineRunRecord]:
        return self._session.query(PipelineRunRecord).filter(
            PipelineRunRecord.symbol == symbol
        ).order_by(PipelineRunRecord.created_at.desc()).limit(limit).all()
```

#### TradeProposalRepository
```python
class TradeProposalRepository(BaseRepository[TradeProposalRecord]):
    def get_by_proposal_id(self, proposal_id: str) -> TradeProposalRecord | None:
        return self._session.query(TradeProposalRecord).filter(
            TradeProposalRecord.proposal_id == proposal_id
        ).first()

    def get_by_run_id(self, run_id: str) -> list[TradeProposalRecord]:
        return self._session.query(TradeProposalRecord).filter(
            TradeProposalRecord.run_id == run_id
        ).all()

    def get_approved(self) -> list[TradeProposalRecord]:
        return self._session.query(TradeProposalRecord).filter(
            TradeProposalRecord.risk_check_result == "approved"
        ).all()
```

#### PaperTradeRepository
```python
class PaperTradeRepository(BaseRepository[PaperTradeRecord]):
    def get_open_positions(self) -> list[PaperTradeRecord]:
        return self._session.query(PaperTradeRecord).filter(
            PaperTradeRecord.status == "open"
        ).all()

    def get_closed_trades(self, limit: int = 100) -> list[PaperTradeRecord]:
        return self._session.query(PaperTradeRecord).filter(
            PaperTradeRecord.status == "closed"
        ).order_by(PaperTradeRecord.closed_at.desc()).limit(limit).all()

    def get_by_proposal_id(self, proposal_id: str) -> PaperTradeRecord | None:
        return self._session.query(PaperTradeRecord).filter(
            PaperTradeRecord.proposal_id == proposal_id
        ).first()

    def get_daily_trades(self, date: date) -> list[PaperTradeRecord]:
        """Get all trades closed on a specific date."""
        start = datetime.combine(date, time.min)
        end = datetime.combine(date, time.max)
        return self._session.query(PaperTradeRecord).filter(
            PaperTradeRecord.closed_at.between(start, end)
        ).all()
```

#### LLMCallRepository
```python
class LLMCallRepository(BaseRepository[LLMCallRecord]):
    def get_by_run_id(self, run_id: str) -> list[LLMCallRecord]:
        return self._session.query(LLMCallRecord).filter(
            LLMCallRecord.run_id == run_id
        ).all()

    def get_cost_by_agent(self, agent_type: str) -> float:
        """Sum total cost for an agent type."""
        result = self._session.query(
            func.sum(LLMCallRecord.cost_usd)
        ).filter(
            LLMCallRecord.agent_type == agent_type
        ).scalar()
        return result or 0.0
```

#### ApprovalRepository
```python
class ApprovalRepository(BaseRepository[ApprovalRecord]):
    def get_pending(self) -> list[ApprovalRecord]:
        return self._session.query(ApprovalRecord).filter(
            ApprovalRecord.status == "pending"
        ).all()

    def get_by_proposal_id(self, proposal_id: str) -> ApprovalRecord | None:
        return self._session.query(ApprovalRecord).filter(
            ApprovalRecord.proposal_id == proposal_id
        ).first()

    def get_expired(self) -> list[ApprovalRecord]:
        """Get approvals that have expired."""
        now = datetime.now(UTC)
        return self._session.query(ApprovalRecord).filter(
            ApprovalRecord.status == "pending",
            ApprovalRecord.expires_at <= now,
        ).all()
```

## Database Setup

**File:** `orchestrator/src/orchestrator/storage/database.py`

```python
def create_db_engine(database_url: str) -> Engine:
    """Create SQLAlchemy engine from URL."""
    return create_engine(database_url, echo=False)

def init_db(engine: Engine) -> None:
    """Create all tables if they don't exist."""
    SQLModel.metadata.create_all(engine)

def get_session(engine: Engine) -> Session:
    """Get a new session."""
    return Session(engine)
```

## Migrations

**File:** `orchestrator/src/orchestrator/storage/migrations.py`

Manual migrations for schema changes:

```python
def migrate_v1_to_v2(session: Session) -> None:
    """Add new column to table."""
    # Alembic not used; manual for simplicity
    with session.begin():
        # Execute raw SQL
        session.exec(
            text("""
            ALTER TABLE paper_trades
            ADD COLUMN IF NOT EXISTS close_reason VARCHAR;
            """)
        )
```

Called on app startup:

```python
from orchestrator.storage.migrations import run_migrations

init_db(engine)
run_migrations(session)
```

## Usage in Application

### In __main__.py

```python
from orchestrator.storage.database import create_db_engine, get_session, init_db
from orchestrator.storage.repository import (
    PipelineRepository,
    TradeProposalRepository,
    PaperTradeRepository,
)

# Setup
engine = create_db_engine(database_url)
init_db(engine)

# Create repositories
with get_session(engine) as session:
    pipeline_repo = PipelineRepository(session)
    trade_repo = PaperTradeRepository(session)
    proposal_repo = TradeProposalRepository(session)

    # Pass to PipelineRunner
    runner = PipelineRunner(
        pipeline_repo=pipeline_repo,
        proposal_repo=proposal_repo,
        ...
    )
```

### In Pipeline

```python
# Store pipeline run
pipeline_repo.create(
    PipelineRunRecord(
        run_id=run_id,
        symbol=symbol,
        status="completed",
    )
)

# Store proposal
proposal_repo.create(
    TradeProposalRecord(
        proposal_id=proposal.proposal_id,
        run_id=run_id,
        proposal_json=proposal.model_dump_json(),
        risk_check_result="approved" if result.approved else "rejected",
        risk_check_reason=result.reason,
    )
)

# Store trade
trade_repo.create(
    PaperTradeRecord(
        trade_id=trade_id,
        proposal_id=proposal.proposal_id,
        symbol=proposal.symbol,
        side=proposal.side.value,
        entry_price=execution_result.entry_price,
        quantity=execution_result.quantity,
    )
)
```

## Configuration

Environment variables:

```python
database_url: str = "sqlite:///data/sentinel.db"  # SQLite path
```

SQLite location:
- **Default:** `orchestrator/data/sentinel.db` (relative to cwd)
- **Absolute:** Set `DATABASE_URL=sqlite:////absolute/path/sentinel.db`

## Querying with Telegram

When user runs `/history`, bot queries trades:

```python
# In TelegramBot.history_handler()
with get_session(engine) as session:
    repo = PaperTradeRepository(session)
    closed_trades = repo.get_closed_trades(limit=10)
    for trade in closed_trades:
        print(f"{trade.symbol} {trade.side} @ {trade.entry_price}: ${trade.pnl:.2f}")
```

## Limitations & Future

Current:
- **SQLite only** — Single-threaded, good for paper trading
- **Manual migrations** — No Alembic
- **No async** — Blocking DB calls (acceptable for current traffic)

Future:
- PostgreSQL support
- Alembic migrations
- Async SQLAlchemy (via asyncio_mode)

## Testing

**File:** `tests/unit/test_repository.py`

Uses in-memory SQLite (`:memory:`):

```python
@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return Session(engine)

def test_trade_repo(session):
    repo = PaperTradeRepository(session)
    trade = repo.create(PaperTradeRecord(...))
    assert repo.get(trade.id) == trade
```

## Related

- [pipeline.md](pipeline.md) — Repository usage in pipeline
- [approval-telegram.md](approval-telegram.md) — Approval record storage
- [configuration.md](configuration.md) — Database configuration
