from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


class PipelineRunRecord(SQLModel, table=True):
    __tablename__ = "pipeline_runs"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(unique=True, index=True)
    symbol: str
    status: str = "running"  # running, completed, failed
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


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
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TradeProposalRecord(SQLModel, table=True):
    __tablename__ = "trade_proposals"

    id: int | None = Field(default=None, primary_key=True)
    proposal_id: str = Field(unique=True, index=True)
    run_id: str = Field(index=True)
    proposal_json: str  # Full JSON
    risk_check_result: str = ""  # approved / rejected
    risk_check_reason: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


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
    opened_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    closed_at: datetime | None = None


class AccountSnapshotRecord(SQLModel, table=True):
    __tablename__ = "account_snapshots"

    id: int | None = Field(default=None, primary_key=True)
    equity: float
    open_positions_count: int = 0
    daily_pnl: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
