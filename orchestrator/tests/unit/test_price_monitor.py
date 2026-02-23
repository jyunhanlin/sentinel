from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.exchange.data_fetcher import TickerSummary
from orchestrator.exchange.paper_engine import CloseResult, PaperEngine
from orchestrator.exchange.price_monitor import PriceMonitor


@pytest.fixture
def monitor():
    engine = MagicMock(spec=PaperEngine)
    data_fetcher = AsyncMock()
    on_close = AsyncMock()
    return PriceMonitor(
        paper_engine=engine,
        data_fetcher=data_fetcher,
        on_close=on_close,
    )


class TestPriceMonitor:
    @pytest.mark.asyncio
    async def test_check_skips_when_no_open_positions(self, monitor):
        monitor._paper_engine.get_open_positions.return_value = []
        await monitor.check()
        monitor._data_fetcher.fetch_current_price.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_fetches_price_per_symbol(self, monitor):
        pos_btc = MagicMock(symbol="BTC/USDT:USDT")
        pos_eth = MagicMock(symbol="ETH/USDT:USDT")
        monitor._paper_engine.get_open_positions.return_value = [pos_btc, pos_eth]
        monitor._paper_engine.check_sl_tp.return_value = []
        monitor._data_fetcher.fetch_current_price.return_value = 68000.0

        await monitor.check()

        # Should fetch price for each unique symbol
        assert monitor._data_fetcher.fetch_current_price.call_count == 2

    @pytest.mark.asyncio
    async def test_check_calls_sl_tp_per_symbol(self, monitor):
        pos = MagicMock(symbol="BTC/USDT:USDT")
        monitor._paper_engine.get_open_positions.return_value = [pos]
        monitor._paper_engine.check_sl_tp.return_value = []
        monitor._data_fetcher.fetch_current_price.return_value = 68000.0

        await monitor.check()

        monitor._paper_engine.check_sl_tp.assert_called_once_with(
            symbol="BTC/USDT:USDT",
            current_price=68000.0,
        )

    @pytest.mark.asyncio
    async def test_check_notifies_on_close(self, monitor):
        pos = MagicMock(symbol="BTC/USDT:USDT")
        monitor._paper_engine.get_open_positions.return_value = [pos]
        monitor._data_fetcher.fetch_current_price.return_value = 66000.0

        close_result = MagicMock(spec=CloseResult)
        close_result.trade_id = "t1"
        close_result.reason = "sl"
        close_result.symbol = "BTC/USDT:USDT"
        close_result.pnl = -50.0
        monitor._paper_engine.check_sl_tp.return_value = [close_result]

        await monitor.check()

        monitor._on_close.assert_called_once_with(close_result)

    @pytest.mark.asyncio
    async def test_check_handles_fetch_error_gracefully(self, monitor):
        pos = MagicMock(symbol="BTC/USDT:USDT")
        monitor._paper_engine.get_open_positions.return_value = [pos]
        monitor._data_fetcher.fetch_current_price.side_effect = Exception("API down")

        # Should not raise, just log
        await monitor.check()
        monitor._paper_engine.check_sl_tp.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_deduplicates_symbols(self, monitor):
        """Two positions on same symbol should only fetch price once."""
        pos1 = MagicMock(symbol="BTC/USDT:USDT")
        pos2 = MagicMock(symbol="BTC/USDT:USDT")
        monitor._paper_engine.get_open_positions.return_value = [pos1, pos2]
        monitor._paper_engine.check_sl_tp.return_value = []
        monitor._data_fetcher.fetch_current_price.return_value = 68000.0

        await monitor.check()

        monitor._data_fetcher.fetch_current_price.assert_called_once_with("BTC/USDT:USDT")
        monitor._paper_engine.check_sl_tp.assert_called_once()


@pytest.fixture
def monitor_with_tick():
    engine = MagicMock(spec=PaperEngine)
    data_fetcher = AsyncMock()
    on_close = AsyncMock()
    on_tick = AsyncMock()
    return PriceMonitor(
        paper_engine=engine,
        data_fetcher=data_fetcher,
        on_close=on_close,
        on_tick=on_tick,
        symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
    )


class TestPriceMonitorOnTick:
    @pytest.mark.asyncio
    async def test_check_calls_on_tick_with_summaries(self, monitor_with_tick):
        monitor = monitor_with_tick
        monitor._paper_engine.get_open_positions.return_value = []
        btc_summary = TickerSummary(symbol="BTC/USDT:USDT", price=69000.0, change_24h_pct=1.2)
        eth_summary = TickerSummary(symbol="ETH/USDT:USDT", price=2530.0, change_24h_pct=-0.5)
        monitor._data_fetcher.fetch_ticker_summary.side_effect = [btc_summary, eth_summary]

        await monitor.check()

        monitor._on_tick.assert_called_once()
        summaries = monitor._on_tick.call_args[0][0]
        assert len(summaries) == 2
        assert summaries[0].symbol == "BTC/USDT:USDT"
        assert summaries[1].symbol == "ETH/USDT:USDT"

    @pytest.mark.asyncio
    async def test_check_skips_on_tick_when_no_callback(self):
        engine = MagicMock(spec=PaperEngine)
        data_fetcher = AsyncMock()
        monitor = PriceMonitor(
            paper_engine=engine,
            data_fetcher=data_fetcher,
            symbols=["BTC/USDT:USDT"],
        )
        engine.get_open_positions.return_value = []
        data_fetcher.fetch_ticker_summary.return_value = TickerSummary(
            symbol="BTC/USDT:USDT", price=69000.0, change_24h_pct=1.0,
        )

        # Should not raise
        await monitor.check()

    @pytest.mark.asyncio
    async def test_on_tick_handles_partial_fetch_failure(self, monitor_with_tick):
        monitor = monitor_with_tick
        monitor._paper_engine.get_open_positions.return_value = []
        btc_summary = TickerSummary(symbol="BTC/USDT:USDT", price=69000.0, change_24h_pct=1.2)
        monitor._data_fetcher.fetch_ticker_summary.side_effect = [
            btc_summary,
            Exception("API down"),
        ]

        await monitor.check()

        # on_tick still called with partial results
        monitor._on_tick.assert_called_once()
        summaries = monitor._on_tick.call_args[0][0]
        assert len(summaries) == 1
        assert summaries[0].symbol == "BTC/USDT:USDT"

    @pytest.mark.asyncio
    async def test_on_tick_uses_symbols_not_positions(self, monitor_with_tick):
        """on_tick fetches all monitored symbols, not just open position symbols."""
        monitor = monitor_with_tick
        # Only BTC position, but both BTC and ETH in symbols
        pos = MagicMock(symbol="BTC/USDT:USDT")
        monitor._paper_engine.get_open_positions.return_value = [pos]
        monitor._paper_engine.check_sl_tp.return_value = []

        btc_summary = TickerSummary(symbol="BTC/USDT:USDT", price=69000.0, change_24h_pct=1.2)
        eth_summary = TickerSummary(symbol="ETH/USDT:USDT", price=2530.0, change_24h_pct=-0.5)
        monitor._data_fetcher.fetch_ticker_summary.side_effect = [btc_summary, eth_summary]
        monitor._data_fetcher.fetch_current_price.return_value = 69000.0

        await monitor.check()

        # on_tick gets ALL symbols (BTC + ETH), not just position symbols
        assert monitor._data_fetcher.fetch_ticker_summary.call_count == 2
