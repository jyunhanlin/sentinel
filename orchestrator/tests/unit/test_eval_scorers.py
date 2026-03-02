import pytest

from orchestrator.eval.dataset import ExpectedMarket, ExpectedProposal
from orchestrator.eval.scorers import RuleScorer, ScoreResult
from orchestrator.models import (
    EntryOrder,
    Momentum,
    Side,
    TakeProfit,
    TechnicalAnalysis,
    TradeProposal,
    Trend,
    VolatilityRegime,
)


class TestRuleScorer:
    def test_score_proposal_side_pass(self):
        scorer = RuleScorer()
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
            position_size_risk_pct=1.0, stop_loss=93000.0,
            take_profit=[TakeProfit(price=97000.0, close_pct=100)],
            time_horizon="4h", confidence=0.7, invalid_if=[], rationale="test",
        )
        expected = ExpectedProposal(side=["long"])
        results = scorer.score_proposal(proposal, expected)
        side_result = next(r for r in results if r.field == "side")
        assert side_result.passed is True

    def test_score_proposal_side_fail(self):
        scorer = RuleScorer()
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
            position_size_risk_pct=1.0, stop_loss=93000.0,
            take_profit=[TakeProfit(price=97000.0, close_pct=100)],
            time_horizon="4h", confidence=0.7, invalid_if=[], rationale="test",
        )
        expected = ExpectedProposal(side=["short", "flat"])
        results = scorer.score_proposal(proposal, expected)
        side_result = next(r for r in results if r.field == "side")
        assert side_result.passed is False

    def test_score_market_trend_pass(self):
        scorer = RuleScorer()
        technical = TechnicalAnalysis(
            label="short_term", trend=Trend.UP, trend_strength=28.0,
            volatility_regime=VolatilityRegime.MEDIUM, volatility_pct=2.5,
            momentum=Momentum.BULLISH, rsi=62.0, key_levels=[], risk_flags=[],
        )
        expected = ExpectedMarket(trend=["up"])
        results = scorer.score_market(technical, expected)
        assert all(r.passed for r in results)

    def test_score_market_trend_fail(self):
        scorer = RuleScorer()
        technical = TechnicalAnalysis(
            label="short_term", trend=Trend.DOWN, trend_strength=30.0,
            volatility_regime=VolatilityRegime.HIGH, volatility_pct=5.0,
            momentum=Momentum.BEARISH, rsi=35.0, key_levels=[], risk_flags=[],
        )
        expected = ExpectedMarket(trend=["up"], volatility_regime=["low", "medium"])
        results = scorer.score_market(technical, expected)
        failed = [r for r in results if not r.passed]
        assert len(failed) == 2  # both trend and volatility fail

    def test_score_result_is_frozen(self):
        result = ScoreResult(field="test", passed=True, expected="x", actual="x")
        with pytest.raises(Exception):
            result.passed = False
