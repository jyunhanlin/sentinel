"""Lightweight schema migration system with version tracking.

Migrations are registered in _REGISTRY by integer version. Each migration
runs inside a transaction and is recorded in the ``schema_migrations`` table
so it is never re-applied.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import structlog
from sqlalchemy import Connection, Engine, inspect, text
from sqlmodel import Field, SQLModel

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Version-tracking table
# ---------------------------------------------------------------------------

class SchemaMigration(SQLModel, table=True):
    __tablename__ = "schema_migrations"

    id: int | None = Field(default=None, primary_key=True)
    version: int = Field(unique=True, index=True)
    name: str
    applied_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Migration registry:  version → (name, migration_fn)
# ---------------------------------------------------------------------------

_REGISTRY: dict[int, tuple[str, Callable[[Connection], None]]] = {}


def _register(version: int, name: str):
    """Decorator to register a migration function."""

    def decorator(fn: Callable[[Connection], None]):
        _REGISTRY[version] = (name, fn)
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Migration 001 — add leverage columns to paper_trades
# ---------------------------------------------------------------------------

_PAPER_TRADES_COLUMNS: list[tuple[str, str, str]] = [
    ("leverage", "INTEGER", "1"),
    ("margin", "REAL", "0.0"),
    ("liquidation_price", "REAL", "0.0"),
    ("close_reason", "TEXT", "''"),
    ("stop_loss", "REAL", "0.0"),
    ("take_profit_json", "TEXT", "'[]'"),
]


@_register(1, "add_leverage_columns")
def migrate_001_add_leverage_columns(conn: Connection) -> None:
    """Add leverage/margin/SL/TP columns to paper_trades (idempotent)."""
    insp = inspect(conn)
    if "paper_trades" not in insp.get_table_names():
        return
    existing = {col["name"] for col in insp.get_columns("paper_trades")}
    for col_name, col_type, default in _PAPER_TRADES_COLUMNS:
        if col_name not in existing:
            stmt = (
                f"ALTER TABLE paper_trades "
                f"ADD COLUMN {col_name} {col_type} DEFAULT {default}"
            )
            conn.execute(text(stmt))
            logger.info("migration_add_column", version=1, column=col_name)


# ---------------------------------------------------------------------------
# Migration 002 — backfill defaults for removed risk_check columns
# ---------------------------------------------------------------------------

_PROPOSAL_DEFAULT_COLUMNS: list[tuple[str, str, str]] = [
    ("risk_check_result", "TEXT", "''"),
    ("risk_check_reason", "TEXT", "''"),
]


@_register(2, "backfill_risk_check_defaults")
def migrate_002_backfill_risk_check_defaults(conn: Connection) -> None:
    """Add DEFAULT to risk_check columns so INSERTs work without them.

    SQLite doesn't support ALTER COLUMN, so we recreate the table with
    defaults.  However, the simpler approach is to just add the columns
    with defaults if they're missing, or — if they already exist — create
    a new table, copy data, and swap.

    Since SQLite ≥3.35 supports UPDATE ... SET DEFAULT, the pragmatic fix
    is to ensure the columns exist with defaults by recreating the table.
    Instead, we take the lightest approach: if the columns exist but lack
    a default, we can't alter them in SQLite.  So we create a new table
    with the correct schema, copy the data, and rename.
    """
    insp = inspect(conn)
    if "trade_proposals" not in insp.get_table_names():
        return

    existing = {col["name"] for col in insp.get_columns("trade_proposals")}
    if "risk_check_result" not in existing:
        # Columns don't exist — nothing to fix
        return

    # Recreate table with defaults on risk_check columns
    conn.execute(text(
        "CREATE TABLE IF NOT EXISTS trade_proposals_new ("
        "  id INTEGER PRIMARY KEY,"
        "  proposal_id TEXT UNIQUE NOT NULL,"
        "  run_id TEXT NOT NULL,"
        "  proposal_json TEXT NOT NULL,"
        "  risk_check_result TEXT DEFAULT '',"
        "  risk_check_reason TEXT DEFAULT '',"
        "  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"
        ")"
    ))
    conn.execute(text(
        "INSERT INTO trade_proposals_new"
        "  (id, proposal_id, run_id, proposal_json,"
        "   risk_check_result, risk_check_reason, created_at) "
        "SELECT id, proposal_id, run_id, proposal_json,"
        "  COALESCE(risk_check_result, ''),"
        "  COALESCE(risk_check_reason, ''), created_at "
        "FROM trade_proposals"
    ))
    conn.execute(text("DROP TABLE trade_proposals"))
    conn.execute(text(
        "ALTER TABLE trade_proposals_new RENAME TO trade_proposals"
    ))
    conn.execute(text(
        "CREATE UNIQUE INDEX IF NOT EXISTS"
        " ix_trade_proposals_proposal_id"
        " ON trade_proposals(proposal_id)"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS"
        " ix_trade_proposals_run_id"
        " ON trade_proposals(run_id)"
    ))
    logger.info("migration_recreate_table", version=2, table="trade_proposals")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _ensure_migration_table(engine: Engine) -> None:
    """Create schema_migrations table if it doesn't exist."""
    SchemaMigration.metadata.create_all(engine, tables=[SchemaMigration.__table__])


def _get_applied_versions(conn: Connection) -> set[int]:
    """Return the set of already-applied migration versions."""
    result = conn.execute(text("SELECT version FROM schema_migrations"))
    return {row[0] for row in result}


def run_migrations(engine: Engine) -> None:
    """Apply all unapplied migrations in version order."""
    _ensure_migration_table(engine)

    with engine.begin() as conn:
        applied = _get_applied_versions(conn)
        for version in sorted(_REGISTRY):
            if version in applied:
                continue
            name, fn = _REGISTRY[version]
            logger.info("migration_start", version=version, name=name)
            fn(conn)
            conn.execute(
                text(
                    "INSERT INTO schema_migrations (version, name, applied_at) "
                    "VALUES (:version, :name, :applied_at)"
                ),
                {"version": version, "name": name, "applied_at": datetime.now(UTC)},
            )
            logger.info("migration_done", version=version, name=name)
