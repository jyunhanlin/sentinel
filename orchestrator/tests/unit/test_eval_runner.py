from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.eval.dataset import (
    EvalCase,
    ExpectedOutputs,
    ExpectedProposal,
)
from orchestrator.eval.runner import EvalReport, EvalRunner


def _make_mock_agents(*, proposal_side="long"):
    """Create mock agents for the 5-agent + proposer pipeline."""
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

    tech_short = AsyncMock()
    tech_short.analyze.return_value = MagicMock(
        output=TechnicalAnalysis(
            label="short_term", trend=Trend.UP, trend_strength=28.0,
            volatility_regime=VolatilityRegime.MEDIUM, volatility_pct=2.5,
            momentum=Momentum.BULLISH, rsi=62.0, key_levels=[], risk_flags=[],
        ),
        degraded=False, llm_calls=[], messages=[],
    )
    tech_long = AsyncMock()
    tech_long.analyze.return_value = MagicMock(
        output=TechnicalAnalysis(
            label="long_term", trend=Trend.UP, trend_strength=25.0,
            volatility_regime=VolatilityRegime.MEDIUM, volatility_pct=2.0,
            momentum=Momentum.BULLISH, rsi=58.0, key_levels=[], risk_flags=[],
        ),
        degraded=False, llm_calls=[], messages=[],
    )
    positioning = AsyncMock()
    positioning.analyze.return_value = MagicMock(
        output=PositioningAnalysis(
            funding_trend="stable", funding_extreme=False, oi_change_pct=0.0,
            retail_bias="neutral", smart_money_bias="neutral", squeeze_risk="none",
            liquidity_assessment="normal", risk_flags=[], confidence=0.5,
        ),
        degraded=False, llm_calls=[], messages=[],
    )
    catalyst = AsyncMock()
    catalyst.analyze.return_value = MagicMock(
        output=CatalystReport(
            upcoming_events=[], active_events=[],
            risk_level="low", recommendation="proceed", confidence=0.5,
        ),
        degraded=False, llm_calls=[], messages=[],
    )
    correlation = AsyncMock()
    correlation.analyze.return_value = MagicMock(
        output=CorrelationAnalysis(
            dxy_trend="stable", dxy_impact="neutral", sp500_regime="neutral",
            btc_dominance_trend="stable", cross_market_alignment="mixed",
            risk_flags=[], confidence=0.5,
        ),
        degraded=False, llm_calls=[], messages=[],
    )
    proposer = AsyncMock()
    side_map = {"long": Side.LONG, "short": Side.SHORT, "flat": Side.FLAT}
    proposer.analyze.return_value = MagicMock(
        output=TradeProposal(
            symbol="BTC/USDT:USDT", side=side_map[proposal_side],
            entry=EntryOrder(type="market"),
            position_size_risk_pct=1.0, stop_loss=93000.0,
            take_profit=[TakeProfit(price=97000.0, close_pct=100)],
            time_horizon="4h", confidence=0.7, invalid_if=[], rationale="test",
        ),
        degraded=False, llm_calls=[], messages=[],
    )

    return {
        "technical_short_agent": tech_short,
        "technical_long_agent": tech_long,
        "positioning_agent": positioning,
        "catalyst_agent": catalyst,
        "correlation_agent": correlation,
        "proposer_agent": proposer,
    }


class TestEvalRunner:
    @pytest.mark.asyncio
    async def test_run_single_passing_case(self):
        case = EvalCase(
            id="bull_breakout",
            description="Strong uptrend",
            snapshot={"symbol": "BTC/USDT:USDT", "current_price": 95000.0, "ohlcv": []},
            expected=ExpectedOutputs(
                proposal=ExpectedProposal(side=["long"]),
            ),
        )

        agents = _make_mock_agents(proposal_side="long")
        runner = EvalRunner(**agents)

        report = await runner.run(cases=[case], dataset_name="test")
        assert isinstance(report, EvalReport)
        assert report.total_cases == 1
        assert report.passed_cases == 1
        assert report.accuracy == 1.0

    @pytest.mark.asyncio
    async def test_run_single_failing_case(self):
        case = EvalCase(
            id="bear_divergence",
            description="Should be short",
            snapshot={"symbol": "BTC/USDT:USDT", "current_price": 95000.0, "ohlcv": []},
            expected=ExpectedOutputs(
                proposal=ExpectedProposal(side=["short"]),
            ),
        )

        agents = _make_mock_agents(proposal_side="long")
        runner = EvalRunner(**agents)

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
