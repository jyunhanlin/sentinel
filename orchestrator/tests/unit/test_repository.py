import json

import pytest
from sqlmodel import Session, SQLModel, create_engine

from orchestrator.storage.models import PaperTradeRecord
from orchestrator.storage.repository import PaperTradeRepository


@pytest.fixture
def repo():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    session = Session(engine)
    return PaperTradeRepository(session)


class TestPaperTradeRepoLeverage:
    def test_save_trade_with_leverage(self, repo):
        record = repo.save_trade(
            trade_id="t1",
            proposal_id="p1",
            symbol="BTC/USDT:USDT",
            side="long",
            entry_price=68000.0,
            quantity=0.1,
            risk_pct=1.0,
            leverage=10,
            margin=680.0,
            liquidation_price=61880.0,
            stop_loss=67000.0,
            take_profit=[70000.0],
        )
        assert record.leverage == 10
        assert record.margin == 680.0
        assert record.liquidation_price == 61880.0
        assert record.stop_loss == 67000.0
        assert json.loads(record.take_profit_json) == [70000.0]

    def test_get_closed_paginated_basic(self, repo):
        for i in range(7):
            repo.save_trade(
                trade_id=f"t{i}",
                proposal_id=f"p{i}",
                symbol="BTC/USDT:USDT",
                side="long",
                entry_price=68000.0,
                quantity=0.1,
            )
            repo.update_trade_closed(f"t{i}", exit_price=69000.0, pnl=100.0, fees=3.4)

        trades, total = repo.get_closed_paginated(offset=0, limit=5)
        assert len(trades) == 5
        assert total == 7

    def test_get_closed_paginated_with_symbol_filter(self, repo):
        repo.save_trade(
            trade_id="t1",
            proposal_id="p1",
            symbol="BTC/USDT:USDT",
            side="long",
            entry_price=68000.0,
            quantity=0.1,
        )
        repo.save_trade(
            trade_id="t2",
            proposal_id="p2",
            symbol="ETH/USDT:USDT",
            side="short",
            entry_price=2400.0,
            quantity=1.0,
        )
        repo.update_trade_closed("t1", exit_price=69000.0, pnl=100.0, fees=3.4)
        repo.update_trade_closed("t2", exit_price=2300.0, pnl=100.0, fees=1.2)

        trades, total = repo.get_closed_paginated(
            offset=0, limit=5, symbol="BTC/USDT:USDT"
        )
        assert len(trades) == 1
        assert total == 1
        assert trades[0].symbol == "BTC/USDT:USDT"

    def test_update_trade_position_modified(self, repo):
        repo.save_trade(
            trade_id="t1",
            proposal_id="p1",
            symbol="BTC/USDT:USDT",
            side="long",
            entry_price=68000.0,
            quantity=0.1,
            leverage=10,
            margin=680.0,
            liquidation_price=61880.0,
        )
        updated = repo.update_trade_position(
            trade_id="t1",
            entry_price=68500.0,
            quantity=0.15,
            margin=1027.5,
            liquidation_price=61700.0,
        )
        assert updated.entry_price == 68500.0
        assert updated.quantity == 0.15
        assert updated.margin == 1027.5

    def test_update_trade_partial_close(self, repo):
        repo.save_trade(
            trade_id="t1",
            proposal_id="p1",
            symbol="BTC/USDT:USDT",
            side="long",
            entry_price=68000.0,
            quantity=0.1,
            leverage=10,
            margin=680.0,
            liquidation_price=61880.0,
        )
        updated = repo.update_trade_partial_close(
            trade_id="t1",
            remaining_qty=0.05,
            remaining_margin=340.0,
        )
        assert updated.quantity == 0.05
        assert updated.margin == 340.0
        assert updated.status == "open"  # still open
