from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.exchange.external_data import ExternalDataFetcher


class TestExternalDataFetcher:
    @pytest.mark.asyncio
    async def test_fetch_dxy_data(self):
        fetcher = ExternalDataFetcher()
        with patch("orchestrator.exchange.external_data.aiohttp") as mock_aiohttp:
            mock_response = AsyncMock()
            mock_response.json.return_value = {
                "chart": {"result": [{"indicators": {"quote": [{"close": [104.0, 104.2, 104.5]}]}}]}
            }
            mock_response.status = 200
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_aiohttp.ClientSession.return_value.__aenter__ = AsyncMock(
                return_value=mock_session,
            )
            mock_aiohttp.ClientSession.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await fetcher.fetch_dxy_data()
            assert "current" in result
            assert "trend_5d" in result

    @pytest.mark.asyncio
    async def test_fetch_dxy_data_fallback_on_error(self):
        fetcher = ExternalDataFetcher()
        with patch("orchestrator.exchange.external_data.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession.side_effect = Exception("Network error")

            result = await fetcher.fetch_dxy_data()
            assert result["current"] == 0.0
            assert result["trend_5d"] == []

    @pytest.mark.asyncio
    async def test_fetch_sp500_data_fallback_on_error(self):
        fetcher = ExternalDataFetcher()
        with patch("orchestrator.exchange.external_data.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession.side_effect = Exception("Network error")

            result = await fetcher.fetch_sp500_data()
            assert result["current"] == 0.0
            assert result["trend_5d"] == []

    @pytest.mark.asyncio
    async def test_fetch_btc_dominance_fallback_on_error(self):
        fetcher = ExternalDataFetcher()
        with patch("orchestrator.exchange.external_data.aiohttp") as mock_aiohttp:
            mock_aiohttp.ClientSession.side_effect = Exception("Network error")

            result = await fetcher.fetch_btc_dominance()
            assert result["current"] == 0.0
            assert result["change_7d"] == 0.0

    @pytest.mark.asyncio
    async def test_fetch_economic_calendar_returns_empty(self):
        fetcher = ExternalDataFetcher()
        result = await fetcher.fetch_economic_calendar()
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_exchange_announcements_returns_empty(self):
        fetcher = ExternalDataFetcher()
        result = await fetcher.fetch_exchange_announcements()
        assert result == []
