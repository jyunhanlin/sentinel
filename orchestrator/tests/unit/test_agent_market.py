from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.market import MarketAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import MarketInterpretation, Trend


def make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        current_price=95200.0,
        volume_24h=1_000_000.0,
        funding_rate=0.0001,
        ohlcv=[
            [1700000000000, 94000.0, 95500.0, 93500.0, 95200.0, 1000.0],
            [1700003600000, 95200.0, 96000.0, 95000.0, 95800.0, 800.0],
        ],
    )


class TestMarketAgent:
    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                '{"trend": "up", "volatility_regime": "medium", '
                '"key_levels": [{"type": "support", "price": 93000}], '
                '"risk_flags": ["funding_elevated"]}'
            ),
            model="test",
            input_tokens=200,
            output_tokens=100,
            latency_ms=1000,
        )

        agent = MarketAgent(client=mock_client)
        result = await agent.analyze(snapshot=make_snapshot())

        assert isinstance(result.output, MarketInterpretation)
        assert result.output.trend == Trend.UP
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_degrade_returns_neutral(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken",
            model="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        agent = MarketAgent(client=mock_client, max_retries=0)
        result = await agent.analyze(snapshot=make_snapshot())

        assert result.degraded is True
        assert result.output.trend == Trend.RANGE
