import pytest
from sqlmodel import Session, SQLModel, create_engine

from orchestrator.storage.models import AccountSnapshotRecord, PaperTradeRecord, PipelineRunRecord
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


class TestAccountSnapshotRepository:
    def test_save_and_get_latest(self, session):
        repo = AccountSnapshotRepository(session)
        repo.save_snapshot(equity=10000.0, open_count=2, daily_pnl=-50.0)
        latest = repo.get_latest()
        assert latest is not None
        assert latest.equity == 10000.0
        assert latest.open_positions_count == 2


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
