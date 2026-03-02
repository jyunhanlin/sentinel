from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.agents.base import AgentResult
from orchestrator.exchange.data_fetcher import DataFetcher, MarketSnapshot
from orchestrator.models import (
    CatalystReport,
    CorrelationAnalysis,
    EntryOrder,
    Momentum,
    PositioningAnalysis,
    Side,
    TechnicalAnalysis,
    TradeProposal,
    Trend,
    VolatilityRegime,
)
from orchestrator.pipeline.runner import PipelineRunner


def make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT:USDT", timeframe="4h",
        current_price=95200.0, volume_24h=1_000_000.0,
        funding_rate=0.0001, ohlcv=[],
    )


def make_technical(label: str = "short_term") -> TechnicalAnalysis:
    return TechnicalAnalysis(
        label=label, trend=Trend.UP, trend_strength=28.0,
        volatility_regime=VolatilityRegime.MEDIUM, volatility_pct=2.3,
        momentum=Momentum.BULLISH, rsi=62.0, key_levels=[], risk_flags=[],
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


def make_proposal() -> TradeProposal:
    return TradeProposal(
        symbol="BTC/USDT:USDT", side=Side.LONG,
        entry=EntryOrder(type="market"),
        position_size_risk_pct=1.0, stop_loss=93000.0,
        take_profit=[], suggested_leverage=10,
        time_horizon="4h", confidence=0.7,
        invalid_if=[], rationale="Test",
    )


def _make_mock_external():
    mock = AsyncMock()
    mock.fetch_dxy_data.return_value = {"current": 0, "change_pct": 0, "trend_5d": []}
    mock.fetch_sp500_data.return_value = {"current": 0, "change_pct": 0, "trend_5d": []}
    mock.fetch_btc_dominance.return_value = {"current": 0, "change_7d": 0}
    mock.fetch_economic_calendar.return_value = []
    mock.fetch_exchange_announcements.return_value = []
    return mock


def _make_mock_fetcher():
    mock = AsyncMock(spec=DataFetcher)
    mock.fetch_snapshot.return_value = make_snapshot()
    mock.fetch_positioning_data.return_value = {
        "funding_rate_history": [], "open_interest": 0, "oi_change_pct": 0,
        "long_short_ratio": 1.0, "top_trader_long_short_ratio": 1.0,
        "order_book_summary": {"bid_depth": 0, "ask_depth": 0},
    }
    mock.fetch_macro_indicators.return_value = {
        "ma_200w": 40000.0, "bull_support_upper": 65000.0, "bull_support_lower": 63000.0,
    }
    return mock


def _make_runner(**overrides):
    mock_tech_short = AsyncMock()
    mock_tech_short.analyze.return_value = AgentResult(output=make_technical("short_term"))
    mock_tech_long = AsyncMock()
    mock_tech_long.analyze.return_value = AgentResult(output=make_technical("long_term"))
    mock_positioning = AsyncMock()
    mock_positioning.analyze.return_value = AgentResult(output=make_positioning())
    mock_catalyst = AsyncMock()
    mock_catalyst.analyze.return_value = AgentResult(output=make_catalyst())
    mock_correlation = AsyncMock()
    mock_correlation.analyze.return_value = AgentResult(output=make_correlation())
    mock_proposer = AsyncMock()
    mock_proposer.analyze.return_value = AgentResult(output=make_proposal())

    defaults = {
        "data_fetcher": _make_mock_fetcher(),
        "technical_short_agent": mock_tech_short,
        "technical_long_agent": mock_tech_long,
        "positioning_agent": mock_positioning,
        "catalyst_agent": mock_catalyst,
        "correlation_agent": mock_correlation,
        "proposer_agent": mock_proposer,
        "external_data_fetcher": _make_mock_external(),
        "pipeline_repo": MagicMock(),
        "llm_call_repo": MagicMock(),
        "proposal_repo": MagicMock(),
    }
    defaults.update(overrides)
    return PipelineRunner(**defaults), defaults


class TestPipelineRunnerV2:
    @pytest.mark.asyncio
    async def test_execute_calls_all_agents_in_parallel(self):
        runner, mocks = _make_runner()
        result = await runner.execute("BTC/USDT:USDT", timeframe="4h")

        # Verify all agents were called
        mocks["technical_short_agent"].analyze.assert_called_once()
        mocks["technical_long_agent"].analyze.assert_called_once()
        mocks["positioning_agent"].analyze.assert_called_once()
        mocks["catalyst_agent"].analyze.assert_called_once()
        mocks["correlation_agent"].analyze.assert_called_once()
        mocks["proposer_agent"].analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_fetches_all_data_sources(self):
        runner, mocks = _make_runner()
        await runner.execute("BTC/USDT:USDT", timeframe="4h")

        fetcher = mocks["data_fetcher"]
        fetcher.fetch_snapshot.assert_called_once()
        fetcher.fetch_positioning_data.assert_called_once()
        fetcher.fetch_macro_indicators.assert_called_once()

        external = mocks["external_data_fetcher"]
        external.fetch_dxy_data.assert_called_once()
        external.fetch_sp500_data.assert_called_once()
        external.fetch_btc_dominance.assert_called_once()
        external.fetch_economic_calendar.assert_called_once()
        external.fetch_exchange_announcements.assert_called_once()

    @pytest.mark.asyncio
    async def test_proposer_receives_all_analysis(self):
        runner, mocks = _make_runner()
        await runner.execute("BTC/USDT:USDT", timeframe="4h")

        # Proposer should have received all analysis outputs
        call_kwargs = mocks["proposer_agent"].analyze.call_args[1]
        assert "technical_short" in call_kwargs
        assert "technical_long" in call_kwargs
        assert "positioning" in call_kwargs
        assert "catalyst" in call_kwargs
        assert "correlation" in call_kwargs

    @pytest.mark.asyncio
    async def test_result_contains_status(self):
        runner, _ = _make_runner()
        result = await runner.execute("BTC/USDT:USDT", timeframe="4h")

        assert result.status == "completed"
        assert result.symbol == "BTC/USDT:USDT"
        assert result.proposal is not None
