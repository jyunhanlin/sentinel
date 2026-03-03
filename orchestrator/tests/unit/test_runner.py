from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.agents.base import AgentResult
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.llm.client import LLMCallResult
from orchestrator.models import (
    CatalystReport,
    CorrelationAnalysis,
    EntryOrder,
    Momentum,
    PositioningAnalysis,
    Side,
    TakeProfit,
    TechnicalAnalysis,
    TradeProposal,
    Trend,
    VolatilityRegime,
)
from orchestrator.pipeline.runner import PipelineResult, PipelineRunner


def _make_proposal(*, side=Side.LONG, risk_pct=1.0):
    return TradeProposal(
        symbol="BTC/USDT:USDT", side=side, entry=EntryOrder(type="market"),
        position_size_risk_pct=risk_pct, stop_loss=93000.0,
        take_profit=[TakeProfit(price=97000.0, close_pct=100)],
        time_horizon="4h", confidence=0.7,
        invalid_if=[], rationale="test",
    )


def _make_technical(label="short_term"):
    return TechnicalAnalysis(
        label=label, trend=Trend.UP, trend_strength=28.0,
        volatility_regime=VolatilityRegime.MEDIUM, volatility_pct=2.3,
        momentum=Momentum.BULLISH, rsi=62.0, key_levels=[], risk_flags=[],
    )


def _make_positioning():
    return PositioningAnalysis(
        funding_trend="stable", funding_extreme=False, oi_change_pct=0.0,
        retail_bias="neutral", smart_money_bias="neutral", squeeze_risk="none",
        liquidity_assessment="normal", risk_flags=[], confidence=0.5,
    )


def _make_catalyst():
    return CatalystReport(
        upcoming_events=[], active_events=[],
        risk_level="low", recommendation="proceed", confidence=0.5,
    )


def _make_correlation():
    return CorrelationAnalysis(
        dxy_trend="stable", dxy_impact="neutral", sp500_regime="neutral",
        btc_dominance_trend="stable", cross_market_alignment="mixed",
        risk_flags=[], confidence=0.5,
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


def make_llm_call() -> LLMCallResult:
    return LLMCallResult(
        content="{}",
        model="test",
        input_tokens=100,
        output_tokens=50,
        latency_ms=500,
    )


def _make_mock_external():
    mock = AsyncMock()
    mock.fetch_dxy_data.return_value = {"current": 0, "change_pct": 0, "trend_5d": []}
    mock.fetch_sp500_data.return_value = {"current": 0, "change_pct": 0, "trend_5d": []}
    mock.fetch_btc_dominance.return_value = {"current": 0, "change_7d": 0}
    mock.fetch_economic_calendar.return_value = []
    mock.fetch_exchange_announcements.return_value = []
    return mock


def _make_mock_fetcher(snapshot=None):
    mock = AsyncMock()
    mock.fetch_snapshot.return_value = snapshot or make_snapshot()
    mock.fetch_positioning_data.return_value = {
        "funding_rate_history": [], "open_interest": 0, "oi_change_pct": 0,
        "long_short_ratio": 1.0, "top_trader_long_short_ratio": 1.0,
        "order_book_summary": {"bid_depth": 0, "ask_depth": 0},
    }
    mock.fetch_macro_indicators.return_value = {
        "ma_200w": 40000.0, "bull_support_upper": 65000.0, "bull_support_lower": 63000.0,
    }
    return mock


def _make_analysis_agents(*, proposal_output=None):
    """Create mock analysis agents returning default outputs."""
    tech_short = AsyncMock()
    tech_short.analyze.return_value = MagicMock(
        output=_make_technical("short_term"), degraded=False, llm_calls=[], messages=[],
    )
    tech_long = AsyncMock()
    tech_long.analyze.return_value = MagicMock(
        output=_make_technical("long_term"), degraded=False, llm_calls=[], messages=[],
    )
    positioning = AsyncMock()
    positioning.analyze.return_value = MagicMock(
        output=_make_positioning(), degraded=False, llm_calls=[], messages=[],
    )
    catalyst = AsyncMock()
    catalyst.analyze.return_value = MagicMock(
        output=_make_catalyst(), degraded=False, llm_calls=[], messages=[],
    )
    correlation = AsyncMock()
    correlation.analyze.return_value = MagicMock(
        output=_make_correlation(), degraded=False, llm_calls=[], messages=[],
    )
    proposer = AsyncMock()
    proposer.analyze.return_value = MagicMock(
        output=proposal_output or _make_proposal(), degraded=False, llm_calls=[], messages=[],
    )
    return {
        "technical_short_agent": tech_short,
        "technical_long_agent": tech_long,
        "positioning_agent": positioning,
        "catalyst_agent": catalyst,
        "correlation_agent": correlation,
        "proposer_agent": proposer,
    }


def _make_runner(*, snapshot=None, proposal_output=None, **extra):
    agents = _make_analysis_agents(proposal_output=proposal_output)
    defaults = {
        "data_fetcher": _make_mock_fetcher(snapshot),
        "external_data_fetcher": _make_mock_external(),
        "pipeline_repo": MagicMock(),
        "llm_call_repo": MagicMock(),
        "proposal_repo": MagicMock(),
        **agents,
    }
    defaults.update(extra)
    return PipelineRunner(**defaults), defaults


class TestPipelineRunner:
    @pytest.mark.asyncio
    async def test_successful_run(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT", side=Side.LONG,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=1.5, stop_loss=93000.0,
            take_profit=[TakeProfit(price=97000.0, close_pct=100)],
            time_horizon="4h", confidence=0.75,
            invalid_if=[], rationale="Bullish",
        )

        # Override proposer with full AgentResult (including llm_calls)
        proposer = AsyncMock()
        proposer.analyze.return_value = AgentResult(
            output=proposal, degraded=False, llm_calls=[make_llm_call()],
        )
        # Also override analysis agents with full AgentResults
        tech_short = AsyncMock()
        tech_short.analyze.return_value = AgentResult(
            output=_make_technical("short_term"), degraded=False, llm_calls=[make_llm_call()],
        )
        tech_long = AsyncMock()
        tech_long.analyze.return_value = AgentResult(
            output=_make_technical("long_term"), degraded=False, llm_calls=[make_llm_call()],
        )

        runner, _ = _make_runner(
            proposer_agent=proposer,
            technical_short_agent=tech_short,
            technical_long_agent=tech_long,
        )

        result = await runner.execute("BTC/USDT:USDT")

        assert isinstance(result, PipelineResult)
        assert result.status == "completed"
        assert result.proposal is not None
        assert result.proposal.side == Side.LONG

    @pytest.mark.asyncio
    async def test_run_with_invalid_proposal(self):
        # Proposal with SL on wrong side
        bad_proposal = TradeProposal(
            symbol="BTC/USDT:USDT", side=Side.LONG,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=1.5, stop_loss=97000.0,  # above price = invalid
            take_profit=[TakeProfit(price=99000.0, close_pct=100)],
            time_horizon="4h", confidence=0.75,
            invalid_if=[], rationale="Bad proposal",
        )

        runner, _ = _make_runner(proposal_output=bad_proposal)

        result = await runner.execute("BTC/USDT:USDT")

        assert result.status == "rejected"
        assert "stop_loss" in result.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_run_with_model_override(self):
        runner, mocks = _make_runner()

        await runner.execute("BTC/USDT:USDT", model_override="anthropic/claude-opus-4-6")

        # Verify model_override was passed to all agents
        for key in ["technical_short_agent", "technical_long_agent", "positioning_agent",
                     "catalyst_agent", "correlation_agent", "proposer_agent"]:
            call_kwargs = mocks[key].analyze.call_args[1]
            assert call_kwargs["model_override"] == "anthropic/claude-opus-4-6"


class TestPipelineRunnerWithPaperEngine:
    """Tests for paper engine integration (no risk checker)."""

    @pytest.mark.asyncio
    async def test_proposal_opens_position(self):
        """Paper engine opens a position for valid proposals."""
        paper_engine = MagicMock()
        paper_engine.open_position.return_value = MagicMock(trade_id="t-001")

        snapshot = MagicMock(symbol="BTC/USDT:USDT", current_price=95000.0)
        runner, _ = _make_runner(
            snapshot=snapshot,
            paper_engine=paper_engine,
        )

        result = await runner.execute("BTC/USDT:USDT")
        assert result.status == "completed"
        paper_engine.open_position.assert_called_once()


class TestPipelineRunnerApproval:
    """Tests for M4 semi-auto approval flow."""

    @pytest.mark.asyncio
    async def test_approval_required_returns_pending(self):
        paper_engine = MagicMock()

        approval_manager = MagicMock()
        approval_manager.create.return_value = MagicMock(
            approval_id="a-001", snapshot_price=95000.0,
        )

        snapshot = MagicMock(symbol="BTC/USDT:USDT", current_price=95000.0)
        runner, _ = _make_runner(
            snapshot=snapshot,
            paper_engine=paper_engine,
            approval_manager=approval_manager,
        )

        result = await runner.execute("BTC/USDT:USDT")
        assert result.status == "pending_approval"
        assert result.approval_id is not None
        approval_manager.create.assert_called_once()
        paper_engine.open_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_approval_manager_auto_executes(self):
        paper_engine = MagicMock()
        paper_engine.open_position.return_value = MagicMock(trade_id="t-001")

        snapshot = MagicMock(symbol="BTC/USDT:USDT", current_price=95000.0)
        runner, _ = _make_runner(
            snapshot=snapshot,
            paper_engine=paper_engine,
        )

        result = await runner.execute("BTC/USDT:USDT")
        assert result.status == "completed"
        paper_engine.open_position.assert_called_once()


class TestSaveLLMCalls:
    def test_saves_full_messages_json(self):
        """_save_llm_calls should store full messages, not placeholder."""
        llm_call_repo = MagicMock()
        runner, _ = _make_runner(llm_call_repo=llm_call_repo)
        mock_result = MagicMock()
        mock_result.llm_calls = [MagicMock(
            content='{"test": true}', model="test", latency_ms=100,
            input_tokens=50, output_tokens=25,
        )]
        mock_result.messages = [{"role": "user", "content": "analyze"}]

        runner._save_llm_calls("run-1", "technical_short", mock_result)

        call_kwargs = llm_call_repo.save_call.call_args[1]
        assert call_kwargs["prompt"] != "(see messages)"
        assert "analyze" in call_kwargs["prompt"]
