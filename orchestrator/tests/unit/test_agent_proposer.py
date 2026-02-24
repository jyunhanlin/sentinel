from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.proposer import ProposerAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import (
    KeyLevel,
    MarketInterpretation,
    SentimentReport,
    Side,
    TradeProposal,
    Trend,
    VolatilityRegime,
)


def make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        current_price=95200.0,
        volume_24h=1_000_000.0,
        funding_rate=0.0001,
        ohlcv=[[1700000000000, 94000.0, 95500.0, 93500.0, 95200.0, 1000.0]],
    )


def make_sentiment() -> SentimentReport:
    return SentimentReport(
        sentiment_score=72,
        key_events=[],
        sources=["market_data"],
        confidence=0.8,
    )


def make_market() -> MarketInterpretation:
    return MarketInterpretation(
        trend=Trend.UP,
        volatility_regime=VolatilityRegime.MEDIUM,
        key_levels=[KeyLevel(type="support", price=93000.0)],
        risk_flags=[],
    )


class TestProposerAgent:
    @pytest.mark.asyncio
    async def test_successful_proposal(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                '{"symbol": "BTC/USDT:USDT", "side": "long", '
                '"entry": {"type": "market"}, "position_size_risk_pct": 1.5, '
                '"stop_loss": 93000, "take_profit": [{"price": 97000, "close_pct": 100}], '
                '"time_horizon": "4h", "confidence": 0.75, '
                '"invalid_if": [], "rationale": "Bullish momentum"}'
            ),
            model="test",
            input_tokens=300,
            output_tokens=150,
            latency_ms=1500,
        )

        agent = ProposerAgent(client=mock_client)
        result = await agent.analyze(
            snapshot=make_snapshot(),
            sentiment=make_sentiment(),
            market=make_market(),
        )

        assert isinstance(result.output, TradeProposal)
        assert result.output.side == Side.LONG
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_prompt_contains_all_inputs(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                '{"symbol": "BTC/USDT:USDT", "side": "flat", '
                '"entry": {"type": "market"}, "position_size_risk_pct": 0, '
                '"stop_loss": null, "take_profit": [], '
                '"time_horizon": "4h", "confidence": 0.5, '
                '"invalid_if": [], "rationale": "No clear signal"}'  # empty take_profit is fine
            ),
            model="test",
            input_tokens=300,
            output_tokens=150,
            latency_ms=1000,
        )

        agent = ProposerAgent(client=mock_client)
        await agent.analyze(
            snapshot=make_snapshot(),
            sentiment=make_sentiment(),
            market=make_market(),
        )

        call_args = mock_client.call.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
        user_msg = messages[-1]["content"]
        # Should contain data from all three inputs
        assert "sentiment_score" in user_msg or "72" in user_msg
        assert "up" in user_msg.lower() or "trend" in user_msg.lower()
        assert "95200" in user_msg

    @pytest.mark.asyncio
    async def test_degrade_returns_flat(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken",
            model="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        agent = ProposerAgent(client=mock_client, max_retries=0)
        result = await agent.analyze(
            snapshot=make_snapshot(),
            sentiment=make_sentiment(),
            market=make_market(),
        )

        assert result.degraded is True
        assert result.output.side == Side.FLAT
