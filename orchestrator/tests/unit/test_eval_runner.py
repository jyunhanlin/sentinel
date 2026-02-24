from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.eval.dataset import (
    EvalCase,
    ExpectedOutputs,
    ExpectedProposal,
)
from orchestrator.eval.runner import EvalReport, EvalRunner


class TestEvalRunner:
    @pytest.mark.asyncio
    async def test_run_single_passing_case(self):
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

        case = EvalCase(
            id="bull_breakout",
            description="Strong uptrend",
            snapshot={"symbol": "BTC/USDT:USDT", "current_price": 95000.0, "ohlcv": []},
            expected=ExpectedOutputs(
                proposal=ExpectedProposal(side=["long"]),
            ),
        )

        # Mock agents to return expected outputs
        sentiment_agent = AsyncMock()
        sentiment_agent.analyze.return_value = MagicMock(
            output=SentimentReport(
                sentiment_score=75, key_events=[], sources=["test"], confidence=0.8
            ),
            degraded=False, llm_calls=[], messages=[],
        )
        market_agent = AsyncMock()
        market_agent.analyze.return_value = MagicMock(
            output=MarketInterpretation(
                trend=Trend.UP, volatility_regime=VolatilityRegime.MEDIUM,
                key_levels=[], risk_flags=[],
            ),
            degraded=False, llm_calls=[], messages=[],
        )
        proposer_agent = AsyncMock()
        proposer_agent.analyze.return_value = MagicMock(
            output=TradeProposal(
                symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
                position_size_risk_pct=1.0, stop_loss=93000.0,
                take_profit=[TakeProfit(price=97000.0, close_pct=100)],
                time_horizon="4h", confidence=0.7, invalid_if=[], rationale="Bullish",
            ),
            degraded=False, llm_calls=[], messages=[],
        )

        runner = EvalRunner(
            sentiment_agent=sentiment_agent,
            market_agent=market_agent,
            proposer_agent=proposer_agent,
        )

        report = await runner.run(cases=[case], dataset_name="test")
        assert isinstance(report, EvalReport)
        assert report.total_cases == 1
        assert report.passed_cases == 1
        assert report.accuracy == 1.0

    @pytest.mark.asyncio
    async def test_run_single_failing_case(self):
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

        case = EvalCase(
            id="bear_divergence",
            description="Should be short",
            snapshot={"symbol": "BTC/USDT:USDT", "current_price": 95000.0, "ohlcv": []},
            expected=ExpectedOutputs(
                proposal=ExpectedProposal(side=["short"]),
            ),
        )

        sentiment_agent = AsyncMock()
        sentiment_agent.analyze.return_value = MagicMock(
            output=SentimentReport(
                sentiment_score=50, key_events=[], sources=[], confidence=0.5
            ),
            degraded=False, llm_calls=[], messages=[],
        )
        market_agent = AsyncMock()
        market_agent.analyze.return_value = MagicMock(
            output=MarketInterpretation(
                trend=Trend.UP, volatility_regime=VolatilityRegime.LOW,
                key_levels=[], risk_flags=[],
            ),
            degraded=False, llm_calls=[], messages=[],
        )
        proposer_agent = AsyncMock()
        proposer_agent.analyze.return_value = MagicMock(
            output=TradeProposal(
                symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
                position_size_risk_pct=1.0, stop_loss=93000.0,
                take_profit=[TakeProfit(price=97000.0, close_pct=100)],
                time_horizon="4h", confidence=0.7, invalid_if=[], rationale="wrong",
            ),
            degraded=False, llm_calls=[], messages=[],
        )

        runner = EvalRunner(
            sentiment_agent=sentiment_agent,
            market_agent=market_agent,
            proposer_agent=proposer_agent,
        )

        report = await runner.run(cases=[case], dataset_name="test")
        assert report.passed_cases == 0
        assert report.failed_cases == 1
        assert len(report.case_results) == 1
        assert report.case_results[0].passed is False

    def test_eval_report_is_frozen(self):
        report = EvalReport(
            dataset_name="test", total_cases=0, passed_cases=0,
            failed_cases=0, accuracy=0.0, case_results=[],
        )
        with pytest.raises(Exception):
            report.total_cases = 5
