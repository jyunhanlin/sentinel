from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.positioning import PositioningAgent
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import PositioningAnalysis

VALID_JSON = (
    '```json\n'
    '{"funding_trend": "rising", "funding_extreme": false, '
    '"oi_change_pct": 3.5, "retail_bias": "long", '
    '"smart_money_bias": "short", "squeeze_risk": "long_squeeze", '
    '"liquidity_assessment": "normal", '
    '"risk_flags": ["funding_elevated"], "confidence": 0.7}\n```'
)


class TestPositioningAgent:
    @pytest.mark.asyncio
    async def test_prompt_contains_skill_and_data(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=VALID_JSON, model="test",
            input_tokens=200, output_tokens=100, latency_ms=1000,
        )

        agent = PositioningAgent(client=mock_client)
        await agent.analyze(
            symbol="BTC/USDT:USDT",
            current_price=95200.0,
            funding_rate_history=[0.0001, 0.0002, 0.0003],
            open_interest=5_000_000_000.0,
            oi_change_pct=3.5,
            long_short_ratio=1.2,
            top_trader_long_short_ratio=0.9,
            order_book_summary={"bid_depth": 100.0, "ask_depth": 120.0},
        )

        messages = mock_client.call.call_args[0][0]
        prompt = messages[0]["content"]
        assert "positioning" in prompt.lower()
        assert "skill" in prompt.lower()
        assert "BTC/USDT:USDT" in prompt

    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=VALID_JSON, model="test",
            input_tokens=200, output_tokens=100, latency_ms=1000,
        )

        agent = PositioningAgent(client=mock_client)
        result = await agent.analyze(
            symbol="BTC/USDT:USDT",
            current_price=95200.0,
            funding_rate_history=[0.0001],
            open_interest=5_000_000_000.0,
            oi_change_pct=3.5,
            long_short_ratio=1.2,
            top_trader_long_short_ratio=0.9,
            order_book_summary={"bid_depth": 100.0, "ask_depth": 120.0},
        )

        assert isinstance(result.output, PositioningAnalysis)
        assert result.output.squeeze_risk == "long_squeeze"
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_degrade_returns_default(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken", model="test",
            input_tokens=100, output_tokens=50, latency_ms=500,
        )

        agent = PositioningAgent(client=mock_client, max_retries=0)
        result = await agent.analyze(
            symbol="BTC/USDT:USDT",
            current_price=95200.0,
            funding_rate_history=[],
            open_interest=0.0,
            oi_change_pct=0.0,
            long_short_ratio=1.0,
            top_trader_long_short_ratio=1.0,
            order_book_summary={"bid_depth": 0.0, "ask_depth": 0.0},
        )

        assert result.degraded is True
        assert result.output.funding_trend == "stable"
