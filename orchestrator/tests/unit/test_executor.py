from unittest.mock import MagicMock

import pytest

from orchestrator.exchange.paper_engine import Position
from orchestrator.execution.executor import ExecutionResult, PaperExecutor
from orchestrator.models import EntryOrder, Side, TakeProfit, TradeProposal


def _make_proposal():
    return TradeProposal(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        entry=EntryOrder(type="market"),
        position_size_risk_pct=1.5,
        stop_loss=93000.0,
        take_profit=[TakeProfit(price=97000.0, close_pct=100)],
        time_horizon="4h",
        confidence=0.75,
        invalid_if=[],
        rationale="test",
    )


class TestPaperExecutor:
    @pytest.mark.asyncio
    async def test_execute_entry(self):
        paper_engine = MagicMock()
        paper_engine.open_position.return_value = Position(
            trade_id="t-001",
            proposal_id="p-001",
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry_price=95200.0,
            quantity=0.075,
            stop_loss=93000.0,
            take_profit=[97000.0],
            opened_at=MagicMock(),
            risk_pct=1.5,
        )
        paper_engine._taker_fee_rate = 0.0005

        executor = PaperExecutor(paper_engine=paper_engine)
        result = await executor.execute_entry(_make_proposal(), current_price=95200.0)

        assert isinstance(result, ExecutionResult)
        assert result.mode == "paper"
        assert result.trade_id == "t-001"
        assert result.entry_price == 95200.0
        paper_engine.open_position.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_sl_tp_is_noop_for_paper(self):
        paper_engine = MagicMock()
        executor = PaperExecutor(paper_engine=paper_engine)
        order_ids = await executor.place_sl_tp(
            symbol="BTC/USDT:USDT",
            side="long",
            quantity=0.075,
            stop_loss=93000.0,
            take_profit=[TakeProfit(price=97000.0, close_pct=100)],
        )
        assert order_ids == []

    @pytest.mark.asyncio
    async def test_execute_entry_passes_leverage(self):
        paper_engine = MagicMock()
        position = MagicMock()
        position.trade_id = "t1"
        position.symbol = "BTC/USDT:USDT"
        position.side = Side.LONG
        position.entry_price = 68000.0
        position.quantity = 0.1
        position.leverage = 10
        position.margin = 680.0
        paper_engine.open_position.return_value = position
        paper_engine._taker_fee_rate = 0.0005

        executor = PaperExecutor(paper_engine=paper_engine)
        proposal = _make_proposal()
        result = await executor.execute_entry(proposal, current_price=68000.0, leverage=10)
        paper_engine.open_position.assert_called_once_with(proposal, 68000.0, leverage=10)
        assert result.mode == "paper"

    def test_execution_result_is_frozen(self):
        result = ExecutionResult(
            trade_id="t-001",
            symbol="BTC/USDT:USDT",
            side="long",
            entry_price=95200.0,
            quantity=0.075,
            fees=3.57,
            mode="paper",
        )
        with pytest.raises(Exception):
            result.mode = "live"
