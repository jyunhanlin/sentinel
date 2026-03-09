from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.critic import CriticAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import (
    CatalystReport,
    CorrelationAnalysis,
    CritiqueResult,
    EntryOrder,
    KeyLevel,
    Momentum,
    PositioningAnalysis,
    Side,
    TakeProfit,
    TechnicalAnalysis,
    TradeProposal,
    Trend,
    VolatilityRegime,
)


def _make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT:USDT", timeframe="4h",
        current_price=95200.0, volume_24h=1_000_000.0,
        funding_rate=0.0001,
        ohlcv=[[1700000000000, 94000.0, 95500.0, 93500.0, 95200.0, 1000.0]],
    )


def _make_proposal() -> TradeProposal:
    return TradeProposal(
        symbol="BTC/USDT:USDT", side=Side.LONG,
        entry=EntryOrder(type="market"),
        position_size_risk_pct=1.5, stop_loss=93000.0,
        take_profit=[TakeProfit(price=97000.0, close_pct=100)],
        time_horizon="4h", confidence=0.75,
        invalid_if=[], rationale="Bullish momentum",
    )


def _make_technical(label="short_term") -> TechnicalAnalysis:
    return TechnicalAnalysis(
        label=label, trend=Trend.UP, trend_strength=28.0,
        volatility_regime=VolatilityRegime.MEDIUM, volatility_pct=2.5,
        momentum=Momentum.BULLISH, rsi=62.0,
        key_levels=[KeyLevel(type="support", price=93000.0)],
        risk_flags=[],
    )


def _make_positioning() -> PositioningAnalysis:
    return PositioningAnalysis(
        funding_trend="stable", funding_extreme=False, oi_change_pct=2.0,
        retail_bias="neutral", smart_money_bias="long", squeeze_risk="none",
        liquidity_assessment="normal", risk_flags=[], confidence=0.7,
    )


def _make_catalyst() -> CatalystReport:
    return CatalystReport(
        upcoming_events=[], active_events=[],
        risk_level="low", recommendation="proceed", confidence=0.8,
    )


def _make_correlation() -> CorrelationAnalysis:
    return CorrelationAnalysis(
        dxy_trend="stable", dxy_impact="neutral",
        sp500_regime="risk_on", btc_dominance_trend="stable",
        cross_market_alignment="favorable", risk_flags=[], confidence=0.7,
    )


def _critic_kwargs():
    return {
        "proposal": _make_proposal(),
        "snapshot": _make_snapshot(),
        "technical_short": _make_technical("short_term"),
        "technical_long": _make_technical("long_term"),
        "positioning": _make_positioning(),
        "catalyst": _make_catalyst(),
        "correlation": _make_correlation(),
    }


PASS_RESPONSE = (
    '```json\n{"verdicts": ['
    '{"dimension": "consistency", "passed": true, "reason": "Side matches bullish analysis"},'
    '{"dimension": "risk_reward", "passed": true, "reason": "R:R of 1.8 is acceptable"},'
    '{"dimension": "input_respect", "passed": true, "reason": "No warnings ignored"},'
    '{"dimension": "parameter_sanity", "passed": true, "reason": "SL/TP within normal range"}'
    '], "overall_passed": true, "suggestions": [], '
    '"summary": "Proposal is consistent and well-structured"}\n```'
)

FAIL_RESPONSE = (
    '```json\n{"verdicts": ['
    '{"dimension": "consistency", "passed": false,'
    ' "reason": "Side is long but momentum is bearish"},'
    '{"dimension": "risk_reward", "passed": true, "reason": "ok"},'
    '{"dimension": "input_respect", "passed": true, "reason": "ok"},'
    '{"dimension": "parameter_sanity", "passed": true, "reason": "ok"}'
    '], "overall_passed": false, '
    '"suggestions": ["Reconsider side given bearish momentum"], '
    '"summary": "Consistency check failed"}\n```'
)


class TestCriticAgent:
    @pytest.mark.asyncio
    async def test_prompt_contains_proposal_and_analysis(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=PASS_RESPONSE, model="test",
            input_tokens=500, output_tokens=200, latency_ms=1500,
        )

        agent = CriticAgent(client=mock_client)
        await agent.analyze(**_critic_kwargs())

        messages = mock_client.call.call_args[0][0]
        prompt = messages[0]["content"]

        assert "critic" in prompt.lower()
        assert "skill" in prompt.lower()
        assert "BTC/USDT:USDT" in prompt
        assert "long" in prompt.lower()       # proposal side
        assert "93000" in prompt               # stop loss
        assert "97000" in prompt               # take profit
        assert "bullish" in prompt.lower()     # momentum from technical

    @pytest.mark.asyncio
    async def test_pass_critique(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=PASS_RESPONSE, model="test",
            input_tokens=500, output_tokens=200, latency_ms=1500,
        )

        agent = CriticAgent(client=mock_client)
        result = await agent.analyze(**_critic_kwargs())

        assert isinstance(result.output, CritiqueResult)
        assert result.output.overall_passed is True
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_fail_critique(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=FAIL_RESPONSE, model="test",
            input_tokens=500, output_tokens=200, latency_ms=1500,
        )

        agent = CriticAgent(client=mock_client)
        result = await agent.analyze(**_critic_kwargs())

        assert isinstance(result.output, CritiqueResult)
        assert result.output.overall_passed is False
        assert len(result.output.suggestions) == 1

    @pytest.mark.asyncio
    async def test_degrade_returns_default_pass(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken", model="test",
            input_tokens=100, output_tokens=50, latency_ms=500,
        )

        agent = CriticAgent(client=mock_client, max_retries=0)
        result = await agent.analyze(**_critic_kwargs())

        assert result.degraded is True
        assert result.output.overall_passed is True  # default: don't block
