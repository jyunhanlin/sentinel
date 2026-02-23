import structlog
from sqlalchemy import Engine, inspect, text
from sqlmodel import Session, SQLModel, create_engine

logger = structlog.get_logger(__name__)


def create_db_engine(database_url: str) -> Engine:
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(database_url, connect_args=connect_args)


# Columns added after initial schema. Each entry: (column_name, SQL type, default).
_PAPER_TRADES_MIGRATIONS: list[tuple[str, str, str]] = [
    ("leverage", "INTEGER", "1"),
    ("margin", "REAL", "0.0"),
    ("liquidation_price", "REAL", "0.0"),
    ("close_reason", "TEXT", "''"),
    ("stop_loss", "REAL", "0.0"),
    ("take_profit_json", "TEXT", "'[]'"),
]


def _migrate_paper_trades(engine: Engine) -> None:
    """Add missing columns to paper_trades table (idempotent)."""
    insp = inspect(engine)
    if "paper_trades" not in insp.get_table_names():
        return
    existing = {col["name"] for col in insp.get_columns("paper_trades")}
    with engine.begin() as conn:
        for col_name, col_type, default in _PAPER_TRADES_MIGRATIONS:
            if col_name not in existing:
                stmt = (
                    f"ALTER TABLE paper_trades "
                    f"ADD COLUMN {col_name} {col_type} DEFAULT {default}"
                )
                conn.execute(text(stmt))
                logger.info("db_migration", action="add_column", column=col_name)


def init_db(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)
    _migrate_paper_trades(engine)


def get_session(engine: Engine) -> Session:
    return Session(engine)
