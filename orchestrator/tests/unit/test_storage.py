import pytest
from sqlmodel import Session, SQLModel, create_engine

from orchestrator.storage.models import PipelineRunRecord
from orchestrator.storage.repository import (
    AccountSnapshotRepository,
    LLMCallRepository,
    PaperTradeRepository,
    PipelineRepository,
    TradeProposalRepository,
)


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
        repo.create_run(run_id="test-001", symbol="BTC/USDT:USDT")
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


class TestLLMCallRepository:
    def test_save_and_list_calls(self, session):
        repo = LLMCallRepository(session)
        repo.save_call(
            run_id="run-001",
            agent_type="sentiment",
            prompt="analyze",
            response='{"score": 72}',
            model="test-model",
            latency_ms=500,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
        )
        repo.save_call(
            run_id="run-001",
            agent_type="market",
            prompt="analyze",
            response='{"trend": "up"}',
            model="test-model",
            latency_ms=600,
            input_tokens=120,
            output_tokens=60,
            cost_usd=0.012,
        )
        calls = repo.list_by_run("run-001")
        assert len(calls) == 2


class TestPaperTradeRepository:
    def test_save_and_get_open(self, session):
        repo = PaperTradeRepository(session)
        repo.save_trade(
            trade_id="t-001",
            proposal_id="p-001",
            symbol="BTC/USDT:USDT",
            side="long",
            entry_price=95000.0,
            quantity=0.075,
            risk_pct=1.5,
        )
        open_positions = repo.get_open_positions()
        assert len(open_positions) == 1
        assert open_positions[0].trade_id == "t-001"
        assert open_positions[0].status == "open"

    def test_close_trade(self, session):
        repo = PaperTradeRepository(session)
        repo.save_trade(
            trade_id="t-002",
            proposal_id="p-002",
            symbol="BTC/USDT:USDT",
            side="long",
            entry_price=95000.0,
            quantity=0.075,
            risk_pct=1.5,
        )
        repo.update_trade_closed(
            trade_id="t-002",
            exit_price=93000.0,
            pnl=-150.0,
            fees=7.13,
        )
        trade = repo.get_by_trade_id("t-002")
        assert trade.status == "closed"
        assert trade.exit_price == 93000.0
        assert trade.pnl == -150.0

    def test_count_consecutive_losses(self, session):
        repo = PaperTradeRepository(session)
        # 3 losses in a row
        for i in range(3):
            repo.save_trade(
                trade_id=f"t-loss-{i}",
                proposal_id=f"p-{i}",
                symbol="BTC/USDT:USDT",
                side="long",
                entry_price=95000.0,
                quantity=0.075,
                risk_pct=1.0,
            )
            repo.update_trade_closed(
                trade_id=f"t-loss-{i}",
                exit_price=93000.0,
                pnl=-150.0,
                fees=7.0,
            )
        assert repo.count_consecutive_losses() == 3

    def test_consecutive_losses_reset_on_win(self, session):
        repo = PaperTradeRepository(session)
        # 1 loss then 1 win
        repo.save_trade(
            trade_id="t-l1", proposal_id="p-1", symbol="BTC/USDT:USDT",
            side="long", entry_price=95000.0, quantity=0.075, risk_pct=1.0,
        )
        repo.update_trade_closed(trade_id="t-l1", exit_price=93000.0, pnl=-150.0, fees=7.0)
        repo.save_trade(
            trade_id="t-w1", proposal_id="p-2", symbol="BTC/USDT:USDT",
            side="long", entry_price=93000.0, quantity=0.08, risk_pct=1.0,
        )
        repo.update_trade_closed(trade_id="t-w1", exit_price=95000.0, pnl=160.0, fees=7.0)
        assert repo.count_consecutive_losses() == 0

    def test_get_daily_pnl(self, session):
        from datetime import UTC, datetime
        repo = PaperTradeRepository(session)
        repo.save_trade(
            trade_id="t-d1", proposal_id="p-1", symbol="BTC/USDT:USDT",
            side="long", entry_price=95000.0, quantity=0.075, risk_pct=1.0,
        )
        repo.update_trade_closed(trade_id="t-d1", exit_price=93000.0, pnl=-150.0, fees=7.0)
        today = datetime.now(UTC).date()
        assert repo.get_daily_pnl(today) == pytest.approx(-150.0)

    def test_get_recent_closed(self, session):
        repo = PaperTradeRepository(session)
        repo.save_trade(
            trade_id="t-r1", proposal_id="p-1", symbol="BTC/USDT:USDT",
            side="long", entry_price=95000.0, quantity=0.075, risk_pct=1.0,
        )
        repo.update_trade_closed(trade_id="t-r1", exit_price=97000.0, pnl=150.0, fees=7.0)
        recent = repo.get_recent_closed(limit=5)
        assert len(recent) == 1
        assert recent[0].trade_id == "t-r1"


class TestPaperTradeRepositoryAllClosed:
    def test_get_all_closed_returns_all(self, session):
        repo = PaperTradeRepository(session)
        # Create 3 closed trades
        for i in range(3):
            repo.save_trade(
                trade_id=f"t-{i}", proposal_id=f"p-{i}",
                symbol="BTC/USDT:USDT", side="long",
                entry_price=95000.0, quantity=0.01,
            )
            repo.update_trade_closed(
                f"t-{i}", exit_price=96000.0, pnl=10.0, fees=1.0
            )
        # Create 1 open trade
        repo.save_trade(
            trade_id="t-open", proposal_id="p-open",
            symbol="BTC/USDT:USDT", side="long",
            entry_price=95000.0, quantity=0.01,
        )
        result = repo.get_all_closed()
        assert len(result) == 3
        assert all(t.status == "closed" for t in result)


class TestAccountSnapshotRepository:
    def test_save_and_get_latest(self, session):
        repo = AccountSnapshotRepository(session)
        repo.save_snapshot(equity=10000.0, open_count=2, daily_pnl=-50.0)
        latest = repo.get_latest()
        assert latest is not None
        assert latest.equity == 10000.0
        assert latest.open_positions_count == 2


class TestAccountSnapshotStatsFields:
    def test_snapshot_has_stats_fields(self, session):
        """AccountSnapshotRecord should have performance stats fields."""
        from orchestrator.storage.models import AccountSnapshotRecord

        snapshot = AccountSnapshotRecord(
            equity=10500.0,
            open_positions_count=2,
            daily_pnl=150.0,
            total_pnl=500.0,
            win_rate=0.625,
            profit_factor=1.85,
            max_drawdown_pct=4.2,
            sharpe_ratio=1.32,
            total_trades=16,
        )
        assert snapshot.total_pnl == 500.0
        assert snapshot.win_rate == 0.625
        assert snapshot.profit_factor == 1.85
        assert snapshot.max_drawdown_pct == 4.2
        assert snapshot.sharpe_ratio == 1.32
        assert snapshot.total_trades == 16


class TestApprovalRecord:
    def test_create_approval_record(self, session):
        from datetime import UTC, datetime, timedelta

        from orchestrator.storage.models import ApprovalRecord

        now = datetime.now(UTC)
        record = ApprovalRecord(
            approval_id="a-001",
            proposal_id="p-001",
            run_id="run-001",
            snapshot_price=95200.0,
            status="pending",
            created_at=now,
            expires_at=now + timedelta(minutes=15),
        )
        assert record.approval_id == "a-001"
        assert record.status == "pending"
        assert record.message_id is None


class TestApprovalRepository:
    def test_save_and_get_approval(self, session):
        from datetime import UTC, datetime, timedelta

        from orchestrator.storage.repository import ApprovalRepository

        repo = ApprovalRepository(session)
        now = datetime.now(UTC)
        record = repo.save_approval(
            approval_id="a-001",
            proposal_id="p-001",
            run_id="run-001",
            snapshot_price=95200.0,
            expires_at=now + timedelta(minutes=15),
        )
        assert record.approval_id == "a-001"
        assert record.status == "pending"

        fetched = repo.get_by_id("a-001")
        assert fetched is not None
        assert fetched.proposal_id == "p-001"

    def test_update_status(self, session):
        from datetime import UTC, datetime, timedelta

        from orchestrator.storage.repository import ApprovalRepository

        repo = ApprovalRepository(session)
        now = datetime.now(UTC)
        repo.save_approval(
            approval_id="a-002",
            proposal_id="p-002",
            run_id="run-002",
            snapshot_price=95000.0,
            expires_at=now + timedelta(minutes=15),
        )
        updated = repo.update_status("a-002", status="approved")
        assert updated.status == "approved"
        assert updated.resolved_at is not None

    def test_get_pending(self, session):
        from datetime import UTC, datetime, timedelta

        from orchestrator.storage.repository import ApprovalRepository

        repo = ApprovalRepository(session)
        now = datetime.now(UTC)
        repo.save_approval(
            approval_id="a-p1",
            proposal_id="p-1",
            run_id="r-1",
            snapshot_price=95000.0,
            expires_at=now + timedelta(minutes=15),
        )
        repo.save_approval(
            approval_id="a-p2",
            proposal_id="p-2",
            run_id="r-2",
            snapshot_price=95000.0,
            expires_at=now + timedelta(minutes=15),
        )
        repo.update_status("a-p2", status="approved")

        pending = repo.get_pending()
        assert len(pending) == 1
        assert pending[0].approval_id == "a-p1"


class TestTradeProposalRepository:
    def test_save_and_get_proposal(self, session):
        repo = TradeProposalRepository(session)
        repo.save_proposal(
            proposal_id="prop-001",
            run_id="run-001",
            proposal_json='{"side": "long"}',
            risk_check_result="approved",
        )
        result = repo.get_latest_by_symbol("run-001")
        assert result is not None

    def test_get_latest_proposals(self, session):
        repo = TradeProposalRepository(session)
        repo.save_proposal(
            proposal_id="prop-001",
            run_id="run-001",
            proposal_json='{"symbol": "BTC/USDT:USDT"}',
        )
        repo.save_proposal(
            proposal_id="prop-002",
            run_id="run-002",
            proposal_json='{"symbol": "ETH/USDT:USDT"}',
        )
        results = repo.get_recent(limit=10)
        assert len(results) == 2
