from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.technical import TechnicalAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import TechnicalAnalysis, Trend, Momentum


def make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT:USDT",
        timeframe="4h",
        current_price=95200.0,
        volume_24h=1_000_000.0,
        funding_rate=0.0001,
        ohlcv=[
            [1700000000000, 94000.0, 95500.0, 93500.0, 95200.0, 1000.0],
            [1700003600000, 95200.0, 96000.0, 95000.0, 95800.0, 800.0],
        ],
    )


VALID_JSON = (
    '```json\n'
    '{"label": "short_term", "trend": "up", "trend_strength": 28.5, '
    '"volatility_regime": "medium", "volatility_pct": 2.3, '
    '"momentum": "bullish", "rsi": 62.0, '
    '"key_levels": [{"type": "support", "price": 93000}], '
    '"risk_flags": []}\n```'
)


class TestTechnicalAgent:
    @pytest.mark.asyncio
    async def test_prompt_contains_skill_name_and_label(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=VALID_JSON, model="test",
            input_tokens=200, output_tokens=100, latency_ms=1000,
        )

        agent = TechnicalAgent(client=mock_client, label="short_term", candle_count=50)
        await agent.analyze(snapshot=make_snapshot())

        messages = mock_client.call.call_args[0][0]
        prompt = messages[0]["content"]
        assert "technical" in prompt.lower()
        assert "skill" in prompt.lower()
        assert "short_term" in prompt

    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=VALID_JSON, model="test",
            input_tokens=200, output_tokens=100, latency_ms=1000,
        )

        agent = TechnicalAgent(client=mock_client, label="short_term", candle_count=50)
        result = await agent.analyze(snapshot=make_snapshot())

        assert isinstance(result.output, TechnicalAnalysis)
        assert result.output.trend == Trend.UP
        assert result.output.momentum == Momentum.BULLISH
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_long_term_includes_macro_data(self):
        mock_client = AsyncMock(spec=LLMClient)
        long_json = (
            '```json\n'
            '{"label": "long_term", "trend": "up", "trend_strength": 30.0, '
            '"volatility_regime": "low", "volatility_pct": 1.2, '
            '"momentum": "bullish", "rsi": 58.0, '
            '"key_levels": [], "risk_flags": [], '
            '"above_200w_ma": true, "bull_support_band_status": "above"}\n```'
        )
        mock_client.call.return_value = LLMCallResult(
            content=long_json, model="test",
            input_tokens=200, output_tokens=100, latency_ms=1000,
        )

        agent = TechnicalAgent(client=mock_client, label="long_term", candle_count=30)
        result = await agent.analyze(
            snapshot=make_snapshot(),
            macro_data={"ma_200w": 42000.0, "bull_support_upper": 68000.0, "bull_support_lower": 65000.0},
        )

        prompt = mock_client.call.call_args[0][0][0]["content"]
        assert "200W MA" in prompt
        assert "42000" in prompt

    @pytest.mark.asyncio
    async def test_degrade_returns_default(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken", model="test",
            input_tokens=100, output_tokens=50, latency_ms=500,
        )

        agent = TechnicalAgent(client=mock_client, label="short_term", candle_count=50, max_retries=0)
        result = await agent.analyze(snapshot=make_snapshot())

        assert result.degraded is True
        assert result.output.trend == Trend.RANGE
        assert result.output.label == "short_term"
