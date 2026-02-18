from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.sentiment import SentimentAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import SentimentReport


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


class TestSentimentAgent:
    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                '{"sentiment_score": 72, "key_events": '
                '[{"event": "BTC rally", "impact": "positive", "source": "market"}], '
                '"sources": ["market_data"], "confidence": 0.75}'
            ),
            model="test",
            input_tokens=200,
            output_tokens=100,
            latency_ms=1000,
        )

        agent = SentimentAgent(client=mock_client)
        result = await agent.analyze(snapshot=make_snapshot())

        assert isinstance(result.output, SentimentReport)
        assert result.output.sentiment_score == 72
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_prompt_contains_market_data(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content='{"sentiment_score": 50, "key_events": [], "sources": [], "confidence": 0.5}',
            model="test",
            input_tokens=200,
            output_tokens=100,
            latency_ms=500,
        )

        agent = SentimentAgent(client=mock_client)
        await agent.analyze(snapshot=make_snapshot())

        call_args = mock_client.call.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
        user_msg = messages[-1]["content"]
        assert "BTC/USDT:USDT" in user_msg
        assert "95200" in user_msg

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

        agent = SentimentAgent(client=mock_client, max_retries=0)
        result = await agent.analyze(snapshot=make_snapshot())

        assert result.degraded is True
        assert result.output.sentiment_score == 50
        assert result.output.confidence <= 0.3
