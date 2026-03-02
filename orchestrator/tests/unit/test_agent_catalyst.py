from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.catalyst import CatalystAgent
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import CatalystReport

VALID_JSON = (
    '```json\n'
    '{"upcoming_events": [{"event": "FOMC Rate Decision", '
    '"time": "2026-03-15T18:00:00Z", "impact": "high", '
    '"direction_bias": "uncertain"}], '
    '"active_events": [], "risk_level": "high", '
    '"recommendation": "wait", "confidence": 0.8}\n```'
)


class TestCatalystAgent:
    @pytest.mark.asyncio
    async def test_prompt_contains_skill_and_data(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=VALID_JSON, model="test",
            input_tokens=200, output_tokens=100, latency_ms=1000,
        )

        agent = CatalystAgent(client=mock_client)
        await agent.analyze(
            symbol="BTC/USDT:USDT",
            current_price=95200.0,
            economic_calendar=[{"event": "FOMC", "time": "2026-03-15T18:00:00Z", "impact": "high"}],
            exchange_announcements=["Binance will delist TOKEN/USDT on 2026-03-20"],
        )

        messages = mock_client.call.call_args[0][0]
        prompt = messages[0]["content"]
        assert "catalyst" in prompt.lower()
        assert "FOMC" in prompt

    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=VALID_JSON, model="test",
            input_tokens=200, output_tokens=100, latency_ms=1000,
        )

        agent = CatalystAgent(client=mock_client)
        result = await agent.analyze(
            symbol="BTC/USDT:USDT",
            current_price=95200.0,
            economic_calendar=[],
            exchange_announcements=[],
        )

        assert isinstance(result.output, CatalystReport)
        assert result.output.recommendation == "wait"
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_degrade_returns_default(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken", model="test",
            input_tokens=100, output_tokens=50, latency_ms=500,
        )

        agent = CatalystAgent(client=mock_client, max_retries=0)
        result = await agent.analyze(
            symbol="BTC/USDT:USDT",
            current_price=95200.0,
            economic_calendar=[],
            exchange_announcements=[],
        )

        assert result.degraded is True
        assert result.output.recommendation == "proceed"
