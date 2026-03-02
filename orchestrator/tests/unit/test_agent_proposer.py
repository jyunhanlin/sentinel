from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.proposer import ProposerAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import (
    CatalystReport,
    CorrelationAnalysis,
    KeyLevel,
    Momentum,
    PositioningAnalysis,
    Side,
    TechnicalAnalysis,
    TradeProposal,
    Trend,
    VolatilityRegime,
)


def make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT:USDT",
        timeframe="4h",
        current_price=95200.0,
        volume_24h=1_000_000.0,
        funding_rate=0.0001,
        ohlcv=[[1700000000000, 94000.0, 95500.0, 93500.0, 95200.0, 1000.0]],
    )


def make_technical(label="short_term") -> TechnicalAnalysis:
    return TechnicalAnalysis(
        label=label, trend=Trend.UP, trend_strength=28.0,
        volatility_regime=VolatilityRegime.MEDIUM, volatility_pct=2.5,
        momentum=Momentum.BULLISH, rsi=62.0,
        key_levels=[KeyLevel(type="support", price=93000.0)],
        risk_flags=[],
    )


def make_positioning() -> PositioningAnalysis:
    return PositioningAnalysis(
        funding_trend="stable", funding_extreme=False, oi_change_pct=2.0,
        retail_bias="neutral", smart_money_bias="long", squeeze_risk="none",
        liquidity_assessment="normal", risk_flags=[], confidence=0.7,
    )


def make_catalyst() -> CatalystReport:
    return CatalystReport(
        upcoming_events=[], active_events=[],
        risk_level="low", recommendation="proceed", confidence=0.8,
    )


def make_correlation() -> CorrelationAnalysis:
    return CorrelationAnalysis(
        dxy_trend="stable", dxy_impact="neutral",
        sp500_regime="risk_on", btc_dominance_trend="stable",
        cross_market_alignment="favorable", risk_flags=[], confidence=0.7,
    )


def _analysis_kwargs():
    return {
        "snapshot": make_snapshot(),
        "technical_short": make_technical("short_term"),
        "technical_long": make_technical("long_term"),
        "positioning": make_positioning(),
        "catalyst": make_catalyst(),
        "correlation": make_correlation(),
    }


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
            content=FLAT_RESPONSE, model="test",
            input_tokens=300, output_tokens=150, latency_ms=1000,
        )

        agent = ProposerAgent(client=mock_client)
        await agent.analyze(**_analysis_kwargs())

        messages = mock_client.call.call_args[0][0]
        prompt = messages[0]["content"]

        # References skill
        assert "proposer" in prompt.lower()
        assert "skill" in prompt.lower()
        # Contains market data
        assert "95200" in prompt
        # Contains short-term technical
        assert "short_term" in prompt
        assert "28.0" in prompt  # ADX
        # Contains long-term technical
        assert "long_term" in prompt
        # Contains positioning
        assert "Positioning" in prompt
        assert "Squeeze Risk" in prompt
        # Contains catalyst
        assert "Catalyst" in prompt
        assert "proceed" in prompt
        # Contains correlation
        assert "Correlation" in prompt
        assert "favorable" in prompt

    @pytest.mark.asyncio
    async def test_successful_proposal(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                '```json\n{"symbol": "BTC/USDT:USDT", "side": "long", '
                '"entry": {"type": "market"}, "position_size_risk_pct": 1.5, '
                '"stop_loss": 93000, "take_profit": [{"price": 97000, "close_pct": 100}], '
                '"time_horizon": "4h", "confidence": 0.75, '
                '"invalid_if": [], "rationale": "Bullish momentum"}\n```'
            ),
            model="test", input_tokens=300, output_tokens=150, latency_ms=1500,
        )

        agent = ProposerAgent(client=mock_client)
        result = await agent.analyze(**_analysis_kwargs())

        assert isinstance(result.output, TradeProposal)
        assert result.output.side == Side.LONG
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_degrade_returns_flat(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken", model="test",
            input_tokens=100, output_tokens=50, latency_ms=500,
        )

        agent = ProposerAgent(client=mock_client, max_retries=0)
        result = await agent.analyze(**_analysis_kwargs())

        assert result.degraded is True
        assert result.output.side == Side.FLAT
