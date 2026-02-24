from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.agents.base import AgentResult
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.llm.client import LLMCallResult
from orchestrator.models import (
    EntryOrder,
    MarketInterpretation,
    SentimentReport,
    Side,
    TakeProfit,
    TradeProposal,
    Trend,
    VolatilityRegime,
)
from orchestrator.pipeline.runner import PipelineResult, PipelineRunner
from orchestrator.risk.checker import RiskResult


def _make_proposal(*, side=Side.LONG, risk_pct=1.0):
    return TradeProposal(
        symbol="BTC/USDT:USDT", side=side, entry=EntryOrder(type="market"),
        position_size_risk_pct=risk_pct, stop_loss=93000.0,
        take_profit=[TakeProfit(price=97000.0, close_pct=100)],
        time_horizon="4h", confidence=0.7,
        invalid_if=[], rationale="test",
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


class TestPipelineRunner:
    @pytest.mark.asyncio
    async def test_successful_run(self):
        sentiment_result = AgentResult(
            output=SentimentReport(
                sentiment_score=72, key_events=[], sources=["test"], confidence=0.8
            ),
            degraded=False,
            llm_calls=[make_llm_call()],
        )
        market_result = AgentResult(
            output=MarketInterpretation(
                trend=Trend.UP,
                volatility_regime=VolatilityRegime.MEDIUM,
                key_levels=[],
                risk_flags=[],
            ),
            degraded=False,
            llm_calls=[make_llm_call()],
        )
        proposal_result = AgentResult(
            output=TradeProposal(
                symbol="BTC/USDT:USDT",
                side=Side.LONG,
                entry=EntryOrder(type="market"),
                position_size_risk_pct=1.5,
                stop_loss=93000.0,
                take_profit=[TakeProfit(price=97000.0, close_pct=100)],
                time_horizon="4h",
                confidence=0.75,
                invalid_if=[],
                rationale="Bullish",
            ),
            degraded=False,
            llm_calls=[make_llm_call()],
        )

        mock_sentiment = AsyncMock()
        mock_sentiment.analyze.return_value = sentiment_result
        mock_market = AsyncMock()
        mock_market.analyze.return_value = market_result
        mock_proposer = AsyncMock()
        mock_proposer.analyze.return_value = proposal_result
        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_snapshot.return_value = make_snapshot()
        mock_repo = MagicMock()
        mock_repo.create_run.return_value = MagicMock(run_id="test-run")

        runner = PipelineRunner(
            data_fetcher=mock_fetcher,
            sentiment_agent=mock_sentiment,
            market_agent=mock_market,
            proposer_agent=mock_proposer,
            pipeline_repo=mock_repo,
            llm_call_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        result = await runner.execute("BTC/USDT:USDT")

        assert isinstance(result, PipelineResult)
        assert result.status == "completed"
        assert result.proposal is not None
        assert result.proposal.side == Side.LONG

    @pytest.mark.asyncio
    async def test_run_with_invalid_proposal(self):
        sentiment_result = AgentResult(
            output=SentimentReport(
                sentiment_score=72, key_events=[], sources=["test"], confidence=0.8
            ),
            degraded=False,
            llm_calls=[make_llm_call()],
        )
        market_result = AgentResult(
            output=MarketInterpretation(
                trend=Trend.UP,
                volatility_regime=VolatilityRegime.MEDIUM,
                key_levels=[],
                risk_flags=[],
            ),
            degraded=False,
            llm_calls=[make_llm_call()],
        )
        # Proposal with SL on wrong side
        proposal_result = AgentResult(
            output=TradeProposal(
                symbol="BTC/USDT:USDT",
                side=Side.LONG,
                entry=EntryOrder(type="market"),
                position_size_risk_pct=1.5,
                stop_loss=97000.0,  # above price = invalid for long
                take_profit=[TakeProfit(price=99000.0, close_pct=100)],
                time_horizon="4h",
                confidence=0.75,
                invalid_if=[],
                rationale="Bad proposal",
            ),
            degraded=False,
            llm_calls=[make_llm_call()],
        )

        mock_sentiment = AsyncMock()
        mock_sentiment.analyze.return_value = sentiment_result
        mock_market = AsyncMock()
        mock_market.analyze.return_value = market_result
        mock_proposer = AsyncMock()
        mock_proposer.analyze.return_value = proposal_result
        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_snapshot.return_value = make_snapshot()
        mock_repo = MagicMock()
        mock_repo.create_run.return_value = MagicMock(run_id="test-run")

        runner = PipelineRunner(
            data_fetcher=mock_fetcher,
            sentiment_agent=mock_sentiment,
            market_agent=mock_market,
            proposer_agent=mock_proposer,
            pipeline_repo=mock_repo,
            llm_call_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        result = await runner.execute("BTC/USDT:USDT")

        assert result.status == "rejected"
        assert "stop_loss" in result.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_run_with_model_override(self):
        sentiment_result = AgentResult(
            output=SentimentReport(
                sentiment_score=50, key_events=[], sources=[], confidence=0.5
            ),
            degraded=False,
            llm_calls=[make_llm_call()],
        )
        market_result = AgentResult(
            output=MarketInterpretation(
                trend=Trend.RANGE,
                volatility_regime=VolatilityRegime.LOW,
                key_levels=[],
                risk_flags=[],
            ),
            degraded=False,
            llm_calls=[make_llm_call()],
        )
        proposal_result = AgentResult(
            output=TradeProposal(
                symbol="BTC/USDT:USDT",
                side=Side.FLAT,
                entry=EntryOrder(type="market"),
                position_size_risk_pct=0,
                stop_loss=None,
                take_profit=[],
                time_horizon="4h",
                confidence=0.5,
                invalid_if=[],
                rationale="No signal",
            ),
            degraded=False,
            llm_calls=[make_llm_call()],
        )

        mock_sentiment = AsyncMock()
        mock_sentiment.analyze.return_value = sentiment_result
        mock_market = AsyncMock()
        mock_market.analyze.return_value = market_result
        mock_proposer = AsyncMock()
        mock_proposer.analyze.return_value = proposal_result
        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_snapshot.return_value = make_snapshot()
        mock_repo = MagicMock()
        mock_repo.create_run.return_value = MagicMock(run_id="test-run")

        runner = PipelineRunner(
            data_fetcher=mock_fetcher,
            sentiment_agent=mock_sentiment,
            market_agent=mock_market,
            proposer_agent=mock_proposer,
            pipeline_repo=mock_repo,
            llm_call_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        await runner.execute("BTC/USDT:USDT", model_override="anthropic/claude-opus-4-6")

        # Verify model_override was passed to all agents
        for mock_agent in [mock_sentiment, mock_market, mock_proposer]:
            call_kwargs = mock_agent.analyze.call_args[1]
            assert call_kwargs["model_override"] == "anthropic/claude-opus-4-6"


class TestPipelineRunnerWithRisk:
    """Tests for M2 risk + paper engine integration."""

    @pytest.mark.asyncio
    async def test_approved_proposal_opens_position(self):
        """When risk check passes, paper engine opens a position."""
        data_fetcher = AsyncMock()
        data_fetcher.fetch_snapshot.return_value = MagicMock(
            symbol="BTC/USDT:USDT",
            current_price=95000.0,
        )

        sentiment_agent = AsyncMock()
        sentiment_agent.analyze.return_value = MagicMock(
            output=MagicMock(), degraded=False, llm_calls=[], messages=[],
        )
        market_agent = AsyncMock()
        market_agent.analyze.return_value = MagicMock(
            output=MagicMock(), degraded=False, llm_calls=[], messages=[],
        )
        proposer_agent = AsyncMock()
        proposer_agent.analyze.return_value = MagicMock(
            output=_make_proposal(side=Side.LONG, risk_pct=1.0),
            degraded=False, llm_calls=[], messages=[],
        )

        risk_checker = MagicMock()
        risk_checker.check.return_value = RiskResult(approved=True)

        paper_engine = MagicMock()

        paper_engine.open_positions_risk_pct = 0.0
        paper_engine.paused = False
        paper_engine.equity = 10000.0
        paper_engine.open_position.return_value = MagicMock(trade_id="t-001")
        paper_engine._trade_repo.get_daily_pnl.return_value = 0.0
        paper_engine._trade_repo.count_consecutive_losses.return_value = 0

        runner = PipelineRunner(
            data_fetcher=data_fetcher,
            sentiment_agent=sentiment_agent,
            market_agent=market_agent,
            proposer_agent=proposer_agent,
            pipeline_repo=MagicMock(),
            llm_call_repo=MagicMock(),
            proposal_repo=MagicMock(),
            risk_checker=risk_checker,
            paper_engine=paper_engine,
        )

        result = await runner.execute("BTC/USDT:USDT")
        assert result.status == "completed"
        paper_engine.open_position.assert_called_once()

    @pytest.mark.asyncio
    async def test_risk_rejected_does_not_open_position(self):
        data_fetcher = AsyncMock()
        data_fetcher.fetch_snapshot.return_value = MagicMock(
            symbol="BTC/USDT:USDT",
            current_price=95000.0,
        )

        sentiment_agent = AsyncMock()
        sentiment_agent.analyze.return_value = MagicMock(
            output=MagicMock(), degraded=False, llm_calls=[], messages=[],
        )
        market_agent = AsyncMock()
        market_agent.analyze.return_value = MagicMock(
            output=MagicMock(), degraded=False, llm_calls=[], messages=[],
        )
        proposer_agent = AsyncMock()
        proposer_agent.analyze.return_value = MagicMock(
            output=_make_proposal(side=Side.LONG, risk_pct=3.0),
            degraded=False, llm_calls=[], messages=[],
        )

        risk_checker = MagicMock()
        risk_checker.check.return_value = RiskResult(
            approved=False, rule_violated="max_single_risk",
            reason="too high", action="reject",
        )

        paper_engine = MagicMock()

        paper_engine.open_positions_risk_pct = 0.0
        paper_engine.paused = False
        paper_engine.equity = 10000.0
        paper_engine._trade_repo.get_daily_pnl.return_value = 0.0
        paper_engine._trade_repo.count_consecutive_losses.return_value = 0

        runner = PipelineRunner(
            data_fetcher=data_fetcher,
            sentiment_agent=sentiment_agent,
            market_agent=market_agent,
            proposer_agent=proposer_agent,
            pipeline_repo=MagicMock(),
            llm_call_repo=MagicMock(),
            proposal_repo=MagicMock(),
            risk_checker=risk_checker,
            paper_engine=paper_engine,
        )

        result = await runner.execute("BTC/USDT:USDT")
        assert result.status == "risk_rejected"
        paper_engine.open_position.assert_not_called()


class TestPipelineRunnerApproval:
    """Tests for M4 semi-auto approval flow."""

    @pytest.mark.asyncio
    async def test_approval_required_returns_pending(self):
        """When approval_manager is set, risk-approved proposals get pending_approval status."""
        data_fetcher = AsyncMock()
        data_fetcher.fetch_snapshot.return_value = MagicMock(
            symbol="BTC/USDT:USDT",
            current_price=95000.0,
        )

        sentiment_agent = AsyncMock()
        sentiment_agent.analyze.return_value = MagicMock(
            output=MagicMock(), degraded=False, llm_calls=[], messages=[],
        )
        market_agent = AsyncMock()
        market_agent.analyze.return_value = MagicMock(
            output=MagicMock(), degraded=False, llm_calls=[], messages=[],
        )
        proposer_agent = AsyncMock()
        proposer_agent.analyze.return_value = MagicMock(
            output=_make_proposal(side=Side.LONG, risk_pct=1.0),
            degraded=False, llm_calls=[], messages=[],
        )
        risk_checker = MagicMock()
        risk_checker.check.return_value = RiskResult(approved=True)

        paper_engine = MagicMock()

        paper_engine.open_positions_risk_pct = 0.0
        paper_engine.equity = 10000.0
        paper_engine._trade_repo.get_daily_pnl.return_value = 0.0
        paper_engine._trade_repo.count_consecutive_losses.return_value = 0

        approval_manager = MagicMock()
        approval_manager.create.return_value = MagicMock(
            approval_id="a-001",
            snapshot_price=95000.0,
        )

        runner = PipelineRunner(
            data_fetcher=data_fetcher,
            sentiment_agent=sentiment_agent,
            market_agent=market_agent,
            proposer_agent=proposer_agent,
            pipeline_repo=MagicMock(),
            llm_call_repo=MagicMock(),
            proposal_repo=MagicMock(),
            risk_checker=risk_checker,
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
        """Without approval_manager, should auto-execute as before."""
        data_fetcher = AsyncMock()
        data_fetcher.fetch_snapshot.return_value = MagicMock(
            symbol="BTC/USDT:USDT",
            current_price=95000.0,
        )
        sentiment_agent = AsyncMock()
        sentiment_agent.analyze.return_value = MagicMock(
            output=MagicMock(), degraded=False, llm_calls=[], messages=[],
        )
        market_agent = AsyncMock()
        market_agent.analyze.return_value = MagicMock(
            output=MagicMock(), degraded=False, llm_calls=[], messages=[],
        )
        proposer_agent = AsyncMock()
        proposer_agent.analyze.return_value = MagicMock(
            output=_make_proposal(side=Side.LONG, risk_pct=1.0),
            degraded=False, llm_calls=[], messages=[],
        )
        risk_checker = MagicMock()
        risk_checker.check.return_value = RiskResult(approved=True)

        paper_engine = MagicMock()

        paper_engine.open_positions_risk_pct = 0.0
        paper_engine.equity = 10000.0
        paper_engine._trade_repo.get_daily_pnl.return_value = 0.0
        paper_engine._trade_repo.count_consecutive_losses.return_value = 0

        runner = PipelineRunner(
            data_fetcher=data_fetcher,
            sentiment_agent=sentiment_agent,
            market_agent=market_agent,
            proposer_agent=proposer_agent,
            pipeline_repo=MagicMock(),
            llm_call_repo=MagicMock(),
            proposal_repo=MagicMock(),
            risk_checker=risk_checker,
            paper_engine=paper_engine,
        )

        result = await runner.execute("BTC/USDT:USDT")
        assert result.status == "completed"
        paper_engine.open_position.assert_called_once()


class TestSaveLLMCalls:
    def test_saves_full_messages_json(self):
        """_save_llm_calls should store full messages, not placeholder."""
        llm_call_repo = MagicMock()
        runner = PipelineRunner(
            data_fetcher=MagicMock(),
            sentiment_agent=MagicMock(),
            market_agent=MagicMock(),
            proposer_agent=MagicMock(),
            pipeline_repo=MagicMock(),
            llm_call_repo=llm_call_repo,
            proposal_repo=MagicMock(),
        )
        mock_result = MagicMock()
        mock_result.llm_calls = [MagicMock(
            content='{"test": true}', model="test", latency_ms=100,
            input_tokens=50, output_tokens=25,
        )]
        mock_result.messages = [{"role": "user", "content": "analyze"}]

        runner._save_llm_calls("run-1", "sentiment", mock_result)

        call_kwargs = llm_call_repo.save_call.call_args[1]
        assert call_kwargs["prompt"] != "(see messages)"
        assert "analyze" in call_kwargs["prompt"]
