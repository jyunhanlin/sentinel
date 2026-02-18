import pytest
from unittest.mock import AsyncMock

from orchestrator.exchange.client import ExchangeClient
from orchestrator.exchange.data_fetcher import DataFetcher, MarketSnapshot


class TestExchangeClient:
    @pytest.mark.asyncio
    async def test_create_client(self):
        client = ExchangeClient(exchange_id="binance")
        assert client.exchange_id == "binance"
        await client.close()

    @pytest.mark.asyncio
    async def test_fetch_ohlcv(self):
        client = ExchangeClient(exchange_id="binance")
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.return_value = [
            [1700000000000, 95000.0, 95500.0, 94500.0, 95200.0, 1000.0],
            [1700000060000, 95200.0, 95800.0, 95100.0, 95600.0, 800.0],
        ]
        client._exchange = mock_exchange

        candles = await client.fetch_ohlcv("BTC/USDT:USDT", "1h", limit=2)
        assert len(candles) == 2
        assert candles[0][1] == 95000.0  # open


class TestDataFetcher:
    @pytest.mark.asyncio
    async def test_fetch_snapshot(self):
        mock_client = AsyncMock(spec=ExchangeClient)
        mock_client.fetch_ohlcv.return_value = [
            [1700000000000, 95000.0, 95500.0, 94500.0, 95200.0, 1000.0],
        ]
        mock_client.fetch_funding_rate.return_value = 0.0001
        mock_client.fetch_ticker.return_value = {
            "last": 95200.0,
            "quoteVolume": 1000000.0,
        }

        fetcher = DataFetcher(mock_client)
        snapshot = await fetcher.fetch_snapshot("BTC/USDT:USDT", timeframe="1h")

        assert snapshot.symbol == "BTC/USDT:USDT"
        assert snapshot.current_price == 95200.0
        assert snapshot.funding_rate == 0.0001
        assert len(snapshot.ohlcv) == 1


class TestMarketSnapshot:
    def test_snapshot_is_immutable(self):
        snapshot = MarketSnapshot(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            current_price=95200.0,
            volume_24h=1000000.0,
            funding_rate=0.0001,
            ohlcv=[[1700000000000, 95000.0, 95500.0, 94500.0, 95200.0, 1000.0]],
        )
        assert snapshot.symbol == "BTC/USDT:USDT"
        with pytest.raises(Exception):
            snapshot.symbol = "ETH"  # type: ignore
