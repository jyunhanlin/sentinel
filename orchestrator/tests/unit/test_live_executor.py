import pytest
from unittest.mock import AsyncMock, MagicMock

from orchestrator.execution.executor import ExecutionResult, LiveExecutor
from orchestrator.models import EntryOrder, Side, TradeProposal


def _make_proposal():
    return TradeProposal(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        entry=EntryOrder(type="market"),
        position_size_risk_pct=1.5,
        stop_loss=93000.0,
        take_profit=[97000.0],
        time_horizon="4h",
        confidence=0.75,
        invalid_if=[],
        rationale="test",
    )


class TestLiveExecutor:
    @pytest.mark.asyncio
    async def test_execute_entry_success(self):
        exchange_client = AsyncMock()
        exchange_client.create_market_order.return_value = {
            "id": "binance-order-001",
            "price": 95200.0,
            "filled": 0.075,
            "fee": {"cost": 3.57},
        }
        position_sizer = MagicMock()
        position_sizer.calculate.return_value = 0.075
        paper_engine = MagicMock()
        paper_engine.equity = 10000.0

        executor = LiveExecutor(
            exchange_client=exchange_client,
            position_sizer=position_sizer,
            paper_engine=paper_engine,
            price_deviation_threshold=0.01,
        )
        result = await executor.execute_entry(
            _make_proposal(), current_price=95200.0
        )

        assert isinstance(result, ExecutionResult)
        assert result.mode == "live"
        assert result.exchange_order_id == "binance-order-001"
        exchange_client.create_market_order.assert_called_once()

    def test_price_deviation_check_raises(self):
        executor = LiveExecutor(
            exchange_client=AsyncMock(),
            position_sizer=MagicMock(),
            paper_engine=MagicMock(),
            price_deviation_threshold=0.01,
        )
        with pytest.raises(ValueError, match="deviated"):
            executor.check_price_deviation(
                snapshot_price=95200.0, current_price=100000.0
            )

    def test_price_deviation_check_within_threshold(self):
        executor = LiveExecutor(
            exchange_client=AsyncMock(),
            position_sizer=MagicMock(),
            paper_engine=MagicMock(),
            price_deviation_threshold=0.01,
        )
        deviation = executor.check_price_deviation(
            snapshot_price=95200.0, current_price=95300.0
        )
        assert deviation < 0.01

    @pytest.mark.asyncio
    async def test_place_sl_tp(self):
        exchange_client = AsyncMock()
        exchange_client.create_stop_order.side_effect = [
            {"id": "sl-001"},
            {"id": "tp-001"},
        ]
        executor = LiveExecutor(
            exchange_client=exchange_client,
            position_sizer=MagicMock(),
            paper_engine=MagicMock(),
            price_deviation_threshold=0.01,
        )
        ids = await executor.place_sl_tp(
            symbol="BTC/USDT:USDT",
            side="long",
            quantity=0.075,
            stop_loss=93000.0,
            take_profit=[97000.0],
        )
        assert len(ids) == 2
        assert "sl-001" in ids
        assert "tp-001" in ids
