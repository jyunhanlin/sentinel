import pytest
from pydantic import ValidationError

from orchestrator.models import (
    EntryOrder,
    KeyEvent,
    KeyLevel,
    MarketInterpretation,
    SentimentReport,
    Side,
    TradeProposal,
    Trend,
    VolatilityRegime,
)


class TestSentimentReport:
    def test_valid_report(self):
        report = SentimentReport(
            sentiment_score=72,
            key_events=[
                KeyEvent(event="BTC ETF inflows", impact="positive", source="Bloomberg")
            ],
            sources=["twitter", "news"],
            confidence=0.8,
        )
        assert report.sentiment_score == 72
        assert len(report.key_events) == 1

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            SentimentReport(
                sentiment_score=101,
                key_events=[],
                sources=[],
                confidence=0.5,
            )

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            SentimentReport(
                sentiment_score=50,
                key_events=[],
                sources=[],
                confidence=1.5,
            )


class TestMarketInterpretation:
    def test_valid_interpretation(self):
        interp = MarketInterpretation(
            trend=Trend.UP,
            volatility_regime=VolatilityRegime.MEDIUM,
            key_levels=[KeyLevel(type="support", price=93000.0)],
            risk_flags=["funding_elevated"],
        )
        assert interp.trend == Trend.UP
        assert len(interp.key_levels) == 1


class TestTradeProposal:
    def test_valid_proposal(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=1.5,
            stop_loss=93000.0,
            take_profit=[95500.0, 97000.0],
            time_horizon="4h",
            confidence=0.75,
            invalid_if=["funding_rate > 0.05%"],
            rationale="Bullish momentum",
        )
        assert proposal.proposal_id is not None
        assert proposal.side == Side.LONG

    def test_flat_side_no_stop_loss_required(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT",
            side=Side.FLAT,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=0.0,
            stop_loss=None,
            take_profit=[],
            time_horizon="4h",
            confidence=0.5,
            invalid_if=[],
            rationale="No trade",
        )
        assert proposal.side == Side.FLAT

    def test_risk_pct_negative_rejected(self):
        with pytest.raises(ValidationError):
            TradeProposal(
                symbol="BTC/USDT:USDT",
                side=Side.LONG,
                entry=EntryOrder(type="market"),
                position_size_risk_pct=-1.0,
                stop_loss=93000.0,
                take_profit=[],
                time_horizon="4h",
                confidence=0.75,
                invalid_if=[],
                rationale="test",
            )
