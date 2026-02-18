import pytest
from sqlmodel import Session, SQLModel, create_engine

from orchestrator.storage.models import PipelineRunRecord
from orchestrator.storage.repository import (
    LLMCallRepository,
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
