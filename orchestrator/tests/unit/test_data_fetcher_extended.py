from unittest.mock import AsyncMock

import pytest

from orchestrator.exchange.client import ExchangeClient
from orchestrator.exchange.data_fetcher import DataFetcher


class TestDataFetcherExtended:
    @pytest.mark.asyncio
    async def test_fetch_positioning_data(self):
        mock_client = AsyncMock(spec=ExchangeClient)
        mock_client.fetch_funding_rate_history.return_value = [0.0001, 0.0002, 0.0003]
        mock_client.fetch_open_interest.return_value = 5_000_000_000.0
        mock_client.fetch_long_short_ratio.return_value = 1.2
        mock_client.fetch_top_trader_long_short_ratio.return_value = 0.9
        mock_client.fetch_order_book.return_value = {"bids": [[95000, 10]], "asks": [[95200, 12]]}

        fetcher = DataFetcher(mock_client)
        result = await fetcher.fetch_positioning_data("BTC/USDT:USDT")

        assert result["funding_rate_history"] == [0.0001, 0.0002, 0.0003]
        assert result["open_interest"] == 5_000_000_000.0
        assert result["long_short_ratio"] == 1.2

    @pytest.mark.asyncio
    async def test_fetch_positioning_data_order_book_depth(self):
        mock_client = AsyncMock(spec=ExchangeClient)
        mock_client.fetch_funding_rate_history.return_value = []
        mock_client.fetch_open_interest.return_value = 0.0
        mock_client.fetch_long_short_ratio.return_value = 1.0
        mock_client.fetch_top_trader_long_short_ratio.return_value = 1.0
        mock_client.fetch_order_book.return_value = {
            "bids": [[95000, 10], [94900, 5]],
            "asks": [[95200, 12], [95300, 8]],
        }

        fetcher = DataFetcher(mock_client)
        result = await fetcher.fetch_positioning_data("BTC/USDT:USDT")

        assert result["order_book_summary"]["bid_depth"] == 15
        assert result["order_book_summary"]["ask_depth"] == 20

    @pytest.mark.asyncio
    async def test_fetch_macro_indicators(self):
        mock_client = AsyncMock(spec=ExchangeClient)
        mock_client.fetch_ohlcv.return_value = [
            [
                i * 604800000, 40000 + i * 100, 41000 + i * 100,
                39000 + i * 100, 40500 + i * 100, 1000,
            ]
            for i in range(210)
        ]

        fetcher = DataFetcher(mock_client)
        result = await fetcher.fetch_macro_indicators("BTC/USDT:USDT")

        assert "ma_200w" in result
        assert "bull_support_upper" in result
        assert "bull_support_lower" in result
        assert result["ma_200w"] > 0
