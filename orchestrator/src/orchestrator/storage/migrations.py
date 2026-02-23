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
