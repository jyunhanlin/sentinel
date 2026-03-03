import json

import pytest
from sqlmodel import Session, SQLModel, create_engine

from orchestrator.storage.repository import TradeProposalRepository


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


class TestTradeProposalRepository:
    def test_get_by_proposal_id_found(self, session: Session):
        repo = TradeProposalRepository(session)
        repo.save_proposal(
            proposal_id="p-123",
            run_id="run-1",
            proposal_json=json.dumps({"symbol": "BTC/USDT:USDT", "side": "long"}),
        )
        result = repo.get_by_proposal_id("p-123")
        assert result is not None
        assert result.proposal_id == "p-123"

    def test_get_by_proposal_id_not_found(self, session: Session):
        repo = TradeProposalRepository(session)
        result = repo.get_by_proposal_id("nonexistent")
        assert result is None
