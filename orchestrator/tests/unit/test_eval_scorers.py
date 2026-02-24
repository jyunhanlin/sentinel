import pytest

from orchestrator.eval.dataset import ExpectedProposal, ExpectedRange, ExpectedSentiment
from orchestrator.eval.scorers import RuleScorer, ScoreResult
from orchestrator.models import (
    EntryOrder,
    SentimentReport,
    Side,
    TakeProfit,
    TradeProposal,
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

    def test_score_sentiment_range_pass(self):
        scorer = RuleScorer()
        sentiment = SentimentReport(
            sentiment_score=75, key_events=[], sources=["test"], confidence=0.8
        )
        expected = ExpectedSentiment(
            sentiment_score=ExpectedRange(min=60, max=90),
            confidence=ExpectedRange(min=0.5),
        )
        results = scorer.score_sentiment(sentiment, expected)
        assert all(r.passed for r in results)

    def test_score_sentiment_range_fail(self):
        scorer = RuleScorer()
        sentiment = SentimentReport(
            sentiment_score=30, key_events=[], sources=["test"], confidence=0.3
        )
        expected = ExpectedSentiment(
            sentiment_score=ExpectedRange(min=60, max=90),
            confidence=ExpectedRange(min=0.5),
        )
        results = scorer.score_sentiment(sentiment, expected)
        failed = [r for r in results if not r.passed]
        assert len(failed) == 2  # both score and confidence fail

    def test_score_result_is_frozen(self):
        result = ScoreResult(field="test", passed=True, expected="x", actual="x")
        with pytest.raises(Exception):
            result.passed = False
