"""Integration test: full 5-agent pipeline with mocked LLM + data."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.agents.catalyst import CatalystAgent
from orchestrator.agents.correlation import CorrelationAgent
from orchestrator.agents.positioning import PositioningAgent
from orchestrator.agents.proposer import ProposerAgent
from orchestrator.agents.technical import TechnicalAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import Side
from orchestrator.pipeline.runner import PipelineResult, PipelineRunner

# -- LLM response fixtures --------------------------------------------------

TECHNICAL_SHORT_RESPONSE = (
    '```json\n'
    '{"label": "short_term", "trend": "up", "trend_strength": 30.0,'
    ' "volatility_regime": "medium", "volatility_pct": 2.5,'
    ' "momentum": "bullish", "rsi": 62.0,'
    ' "key_levels": [{"type": "support", "price": 93000.0}],'
    ' "risk_flags": []}\n```'
)

TECHNICAL_LONG_RESPONSE = (
    '```json\n'
    '{"label": "long_term", "trend": "up", "trend_strength": 25.0,'
    ' "volatility_regime": "medium", "volatility_pct": 2.0,'
    ' "momentum": "bullish", "rsi": 58.0,'
    ' "key_levels": [{"type": "support", "price": 90000.0}],'
    ' "risk_flags": []}\n```'
)

POSITIONING_RESPONSE = (
    '```json\n'
    '{"funding_trend": "stable", "funding_extreme": false,'
    ' "oi_change_pct": 2.0, "retail_bias": "neutral",'
    ' "smart_money_bias": "long", "squeeze_risk": "none",'
    ' "liquidity_assessment": "normal", "risk_flags": [],'
    ' "confidence": 0.7}\n```'
)

CATALYST_RESPONSE = (
    '```json\n'
    '{"upcoming_events": [], "active_events": [],'
    ' "risk_level": "low", "recommendation": "proceed",'
    ' "confidence": 0.8}\n```'
)

CORRELATION_RESPONSE = (
    '```json\n'
    '{"dxy_trend": "weakening", "dxy_impact": "tailwind",'
    ' "sp500_regime": "risk_on", "btc_dominance_trend": "stable",'
    ' "cross_market_alignment": "favorable", "risk_flags": [],'
    ' "confidence": 0.7}\n```'
)

PROPOSER_LONG_RESPONSE = (
    '```json\n'
    '{"symbol": "BTC/USDT:USDT", "side": "long",'
    ' "entry": {"type": "market"}, "position_size_risk_pct": 1.5,'
    ' "stop_loss": 93000.0,'
    ' "take_profit": [{"price": 97000.0, "close_pct": 50},'
    '                  {"price": 99000.0, "close_pct": 100}],'
    ' "suggested_leverage": 10, "time_horizon": "4h",'
    ' "confidence": 0.75, "invalid_if": ["BTC < 92000"],'
    ' "rationale": "Bullish trend across timeframes with favorable macro"}\n```'
)


def _make_llm_result(content: str) -> LLMCallResult:
    return LLMCallResult(
        content=content, model="test",
        input_tokens=200, output_tokens=100, latency_ms=500,
    )


def _make_mock_client(responses: list[str]) -> LLMClient:
    """Create a mock LLMClient that returns responses in order."""
    mock = AsyncMock(spec=LLMClient)
    mock.call.side_effect = [_make_llm_result(r) for r in responses]
    return mock


def _make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT:USDT",
        timeframe="4h",
        current_price=95200.0,
        volume_24h=25_000_000_000.0,
        funding_rate=0.0001,
        ohlcv=[
            [1708300800000, 93000.0, 93800.0, 92500.0, 93500.0, 12000.0],
            [1708315200000, 93500.0, 94200.0, 93200.0, 94100.0, 13500.0],
            [1708329600000, 94100.0, 95000.0, 93800.0, 94800.0, 15000.0],
            [1708344000000, 94800.0, 95500.0, 94500.0, 95400.0, 16000.0],
            [1708358400000, 95400.0, 96800.0, 95200.0, 95200.0, 18000.0],
        ],
    )


def _make_data_fetcher(snapshot: MarketSnapshot | None = None) -> AsyncMock:
    mock = AsyncMock()
    mock.fetch_snapshot.return_value = snapshot or _make_snapshot()
    mock.fetch_positioning_data.return_value = {
        "funding_rate_history": [0.0001, 0.0002, 0.00015],
        "open_interest": 500_000_000,
        "oi_change_pct": 2.0,
        "long_short_ratio": 1.1,
        "top_trader_long_short_ratio": 1.2,
        "order_book_summary": {"bid_depth": 1000, "ask_depth": 900},
    }
    mock.fetch_macro_indicators.return_value = {
        "ma_200w": 40000.0,
        "bull_support_upper": 65000.0,
        "bull_support_lower": 63000.0,
    }
    return mock


def _make_external_data_fetcher() -> AsyncMock:
    mock = AsyncMock()
    mock.fetch_dxy_data.return_value = {
        "current": 103.5, "change_pct": -0.3, "trend_5d": [104.0, 103.8, 103.5],
    }
    mock.fetch_sp500_data.return_value = {
        "current": 5200.0, "change_pct": 0.5, "trend_5d": [5150.0, 5180.0, 5200.0],
    }
    mock.fetch_btc_dominance.return_value = {"current": 52.0, "change_7d": 0.5}
    mock.fetch_economic_calendar.return_value = []
    mock.fetch_exchange_announcements.return_value = []
    return mock


class TestFullPipelineIntegration:
    """End-to-end test with real agents but mocked LLM backend."""

    @pytest.mark.asyncio
    async def test_bullish_pipeline_produces_long_proposal(self):
        """Full pipeline with bullish inputs produces a LONG proposal."""
        # Each agent gets its own LLM client (6 agents, 6 calls)
        responses = [
            TECHNICAL_SHORT_RESPONSE,
            TECHNICAL_LONG_RESPONSE,
            POSITIONING_RESPONSE,
            CATALYST_RESPONSE,
            CORRELATION_RESPONSE,
            PROPOSER_LONG_RESPONSE,
        ]

        # Create a shared mock client that serves all responses
        mock_client = _make_mock_client(responses)

        # Create real agents with mock LLM client
        technical_short = TechnicalAgent(
            client=mock_client, label="short_term", candle_count=50,
        )
        technical_long = TechnicalAgent(
            client=mock_client, label="long_term", candle_count=30,
        )
        positioning = PositioningAgent(client=mock_client)
        catalyst = CatalystAgent(client=mock_client)
        correlation = CorrelationAgent(client=mock_client)
        proposer = ProposerAgent(client=mock_client)

        runner = PipelineRunner(
            data_fetcher=_make_data_fetcher(),
            technical_short_agent=technical_short,
            technical_long_agent=technical_long,
            positioning_agent=positioning,
            catalyst_agent=catalyst,
            correlation_agent=correlation,
            proposer_agent=proposer,
            external_data_fetcher=_make_external_data_fetcher(),
            pipeline_repo=MagicMock(),
            llm_call_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        result = await runner.execute("BTC/USDT:USDT")

        # Verify pipeline completed successfully
        assert isinstance(result, PipelineResult)
        assert result.status == "completed"
        assert result.proposal is not None
        assert result.proposal.side == Side.LONG
        assert result.proposal.stop_loss == 93000.0
        assert len(result.proposal.take_profit) == 2
        assert result.proposal.confidence == 0.75

        # Verify analysis outputs are populated
        assert result.technical_short is not None
        assert result.technical_short.trend.value == "up"
        assert result.technical_long is not None
        assert result.positioning is not None
        assert result.catalyst is not None
        assert result.correlation is not None

        # Verify no degradation
        assert result.technical_short_degraded is False
        assert result.technical_long_degraded is False
        assert result.positioning_degraded is False
        assert result.catalyst_degraded is False
        assert result.correlation_degraded is False
        assert result.proposer_degraded is False

        # All 6 agents should have been called
        assert mock_client.call.call_count == 6

    @pytest.mark.asyncio
    async def test_degraded_agent_still_completes_pipeline(self):
        """If one analysis agent returns garbage, pipeline still completes."""
        responses = [
            "not valid json at all",  # technical_short degrades
            TECHNICAL_LONG_RESPONSE,
            POSITIONING_RESPONSE,
            CATALYST_RESPONSE,
            CORRELATION_RESPONSE,
            PROPOSER_LONG_RESPONSE,
        ]

        mock_client = _make_mock_client(responses)

        technical_short = TechnicalAgent(
            client=mock_client, label="short_term", candle_count=50,
            max_retries=0,
        )
        technical_long = TechnicalAgent(
            client=mock_client, label="long_term", candle_count=30,
        )
        positioning = PositioningAgent(client=mock_client)
        catalyst = CatalystAgent(client=mock_client)
        correlation = CorrelationAgent(client=mock_client)
        proposer = ProposerAgent(client=mock_client)

        runner = PipelineRunner(
            data_fetcher=_make_data_fetcher(),
            technical_short_agent=technical_short,
            technical_long_agent=technical_long,
            positioning_agent=positioning,
            catalyst_agent=catalyst,
            correlation_agent=correlation,
            proposer_agent=proposer,
            external_data_fetcher=_make_external_data_fetcher(),
            pipeline_repo=MagicMock(),
            llm_call_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        result = await runner.execute("BTC/USDT:USDT")

        assert result.status == "completed"
        assert result.technical_short_degraded is True
        # technical_short should have a default/degraded output
        assert result.technical_short is not None
        assert "analysis_degraded" in result.technical_short.risk_flags

