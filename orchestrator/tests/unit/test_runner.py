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
    TradeProposal,
    Trend,
    VolatilityRegime,
)
from orchestrator.pipeline.runner import PipelineResult, PipelineRunner


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
                take_profit=[97000.0],
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
                take_profit=[99000.0],
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
