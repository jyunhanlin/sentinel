from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.correlation import CorrelationAgent
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import CorrelationAnalysis

VALID_JSON = (
    '```json\n'
    '{"dxy_trend": "strengthening", "dxy_impact": "headwind", '
    '"sp500_regime": "risk_off", "btc_dominance_trend": "rising", '
    '"cross_market_alignment": "unfavorable", '
    '"risk_flags": ["dxy_headwind"], "confidence": 0.7}\n```'
)


class TestCorrelationAgent:
    @pytest.mark.asyncio
    async def test_prompt_contains_skill_and_data(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=VALID_JSON, model="test",
            input_tokens=200, output_tokens=100, latency_ms=1000,
        )

        agent = CorrelationAgent(client=mock_client)
        await agent.analyze(
            symbol="BTC/USDT:USDT",
            dxy_data={
                "current": 104.5, "change_pct": 0.3,
                "trend_5d": [103.8, 104.0, 104.2, 104.3, 104.5],
            },
            sp500_data={
                "current": 5800.0, "change_pct": -1.2,
                "trend_5d": [5900, 5880, 5850, 5820, 5800],
            },
            btc_dominance={"current": 54.2, "change_7d": 1.5},
        )

        messages = mock_client.call.call_args[0][0]
        prompt = messages[0]["content"]
        assert "correlation" in prompt.lower()
        assert "DXY" in prompt

    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=VALID_JSON, model="test",
            input_tokens=200, output_tokens=100, latency_ms=1000,
        )

        agent = CorrelationAgent(client=mock_client)
        result = await agent.analyze(
            symbol="BTC/USDT:USDT",
            dxy_data={"current": 104.5, "change_pct": 0.3, "trend_5d": []},
            sp500_data={"current": 5800.0, "change_pct": -1.2, "trend_5d": []},
            btc_dominance={"current": 54.2, "change_7d": 1.5},
        )

        assert isinstance(result.output, CorrelationAnalysis)
        assert result.output.dxy_impact == "headwind"
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_degrade_returns_default(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken", model="test",
            input_tokens=100, output_tokens=50, latency_ms=500,
        )

        agent = CorrelationAgent(client=mock_client, max_retries=0)
        result = await agent.analyze(
            symbol="BTC/USDT:USDT",
            dxy_data={"current": 0, "change_pct": 0, "trend_5d": []},
            sp500_data={"current": 0, "change_pct": 0, "trend_5d": []},
            btc_dominance={"current": 0, "change_7d": 0},
        )

        assert result.degraded is True
        assert result.output.cross_market_alignment == "mixed"
