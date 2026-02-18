import pytest
from sqlmodel import Session, SQLModel, create_engine

from orchestrator.storage.models import PipelineRunRecord
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
