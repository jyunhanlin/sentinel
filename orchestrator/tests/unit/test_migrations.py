from sqlalchemy import inspect, text
from sqlmodel import SQLModel, create_engine

from orchestrator.storage.database import init_db
from orchestrator.storage.migrations import run_migrations


def _create_old_paper_trades(engine):
    """Create a paper_trades table without the new leverage columns."""
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE paper_trades ("
            "  id INTEGER PRIMARY KEY,"
            "  trade_id TEXT UNIQUE,"
            "  proposal_id TEXT,"
            "  symbol TEXT,"
            "  side TEXT,"
            "  entry_price REAL,"
            "  exit_price REAL,"
            "  quantity REAL,"
            "  pnl REAL DEFAULT 0,"
            "  fees REAL DEFAULT 0,"
            "  risk_pct REAL DEFAULT 0,"
            "  status TEXT DEFAULT 'open',"
            "  mode TEXT DEFAULT 'paper',"
            "  exchange_order_id TEXT DEFAULT '',"
            "  sl_order_id TEXT DEFAULT '',"
            "  tp_order_id TEXT DEFAULT '',"
            "  opened_at TIMESTAMP,"
            "  closed_at TIMESTAMP"
            ")"
        ))


_EXPECTED_NEW_COLUMNS = [
    "leverage", "margin", "liquidation_price",
    "close_reason", "stop_loss", "take_profit_json",
]


class TestMigrateOldDb:
    def test_migrate_adds_missing_columns(self):
        """init_db should add leverage columns to an old paper_trades table."""
        engine = create_engine("sqlite:///:memory:")
        _create_old_paper_trades(engine)

        # Verify old schema lacks the new columns
        old_cols = {c["name"] for c in inspect(engine).get_columns("paper_trades")}
        assert "leverage" not in old_cols

        init_db(engine)

        new_cols = {c["name"] for c in inspect(engine).get_columns("paper_trades")}
        for col in _EXPECTED_NEW_COLUMNS:
            assert col in new_cols, f"Missing column: {col}"


class TestIdempotency:
    def test_migrations_are_idempotent(self):
        """Running init_db twice should not error."""
        engine = create_engine("sqlite:///:memory:")
        init_db(engine)
        init_db(engine)  # should not raise

    def test_run_migrations_twice_on_old_db(self):
        """Running migrations twice on an old DB should not error."""
        engine = create_engine("sqlite:///:memory:")
        _create_old_paper_trades(engine)
        SQLModel.metadata.create_all(engine)
        run_migrations(engine)
        run_migrations(engine)  # should not raise


class TestVersionTracking:
    def test_version_tracked_in_schema_migrations(self):
        """Applied migrations are recorded in schema_migrations."""
        engine = create_engine("sqlite:///:memory:")
        _create_old_paper_trades(engine)
        SQLModel.metadata.create_all(engine)
        run_migrations(engine)

        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT version, name FROM schema_migrations ORDER BY version")
            ).fetchall()
        assert len(rows) >= 1
        assert rows[0][0] == 1
        assert rows[0][1] == "add_leverage_columns"

    def test_only_unapplied_migrations_run(self):
        """Already-applied versions are skipped on subsequent runs."""
        engine = create_engine("sqlite:///:memory:")
        _create_old_paper_trades(engine)
        SQLModel.metadata.create_all(engine)
        run_migrations(engine)

        # Record count after first run
        with engine.connect() as conn:
            count_before = conn.execute(
                text("SELECT COUNT(*) FROM schema_migrations")
            ).scalar()

        run_migrations(engine)

        with engine.connect() as conn:
            count_after = conn.execute(
                text("SELECT COUNT(*) FROM schema_migrations")
            ).scalar()

        assert count_after == count_before
