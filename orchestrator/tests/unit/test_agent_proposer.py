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
        volatility_pct=2.5,
        key_levels=[KeyLevel(type="support", price=93000.0)],
        risk_flags=[],
    )


FLAT_RESPONSE = (
    "## Analysis\nNo clear edge.\n\n"
    '```json\n{"symbol": "BTC/USDT:USDT", "side": "flat", '
    '"entry": {"type": "market"}, "position_size_risk_pct": 0, '
    '"stop_loss": null, "take_profit": [], '
    '"time_horizon": "4h", "confidence": 0.5, '
    '"invalid_if": [], "rationale": "No signal"}\n```'
)


class TestProposerAgent:
    @pytest.mark.asyncio
    async def test_prompt_references_skill_and_contains_all_inputs(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=FLAT_RESPONSE,
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

        assert len(messages) == 1
        assert messages[0]["role"] == "user"

        prompt = messages[0]["content"]
        # References skill
        assert "proposer" in prompt.lower()
        assert "skill" in prompt.lower()
        # Contains data from all three inputs
        assert "95200" in prompt
        assert "72" in prompt  # sentiment score
        assert "up" in prompt.lower()  # trend
        assert "2.5" in prompt  # volatility_pct

    @pytest.mark.asyncio
    async def test_successful_proposal(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                "## Trade Analysis\nBullish setup.\n\n"
                '```json\n{"symbol": "BTC/USDT:USDT", "side": "long", '
                '"entry": {"type": "market"}, "position_size_risk_pct": 1.5, '
                '"stop_loss": 93000, "take_profit": [{"price": 97000, "close_pct": 100}], '
                '"time_horizon": "4h", "confidence": 0.75, '
                '"invalid_if": [], "rationale": "Bullish momentum"}\n```'
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
