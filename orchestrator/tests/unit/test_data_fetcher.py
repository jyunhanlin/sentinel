from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orchestrator.exchange.data_fetcher import DataFetcher, TickerSummary


class TestFetchTickerSummary:
    @pytest.mark.asyncio
    async def test_returns_ticker_summary(self):
        client = AsyncMock()
        client.fetch_ticker.return_value = {
            "last": 69123.5,
            "percentage": 1.2,
        }
        fetcher = DataFetcher(client)

        result = await fetcher.fetch_ticker_summary("BTC/USDT:USDT")

        assert isinstance(result, TickerSummary)
        assert result.symbol == "BTC/USDT:USDT"
        assert result.price == 69123.5
        assert result.change_24h_pct == 1.2

    @pytest.mark.asyncio
    async def test_handles_missing_percentage(self):
        client = AsyncMock()
        client.fetch_ticker.return_value = {
            "last": 2530.0,
        }
        fetcher = DataFetcher(client)

        result = await fetcher.fetch_ticker_summary("ETH/USDT:USDT")

        assert result.price == 2530.0
        assert result.change_24h_pct == 0.0
